import asyncio
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
from typing import List, Dict

from projectairsim import ProjectAirSimClient, World, Rover, Drone
from projectairsim.utils import projectairsim_log, unpack_image
from projectairsim.types import ImageType
from .rover_hardcoded_trajectories import *

class AirSimWrapperNode(Node):
    def __init__(self):
        super().__init__('airsim_wrapper_node')
        
        # Declare ROS2 parameters
        self.declare_parameter('client_address', '172.25.240.1')
        self.declare_parameter('scene_config_name', 'scene_car_following1.jsonc')
        self.declare_parameter('sim_config_path', 'src/p_airsim_wrapper/config/sim_config')
        self.declare_parameter('drone_names', ['Drone'])
        self.declare_parameter('sensor_publish_rate', 20.0)  # Hz
        
        # Get parameter values
        self.client_address = self.get_parameter('client_address').value
        self.scene_config_name = self.get_parameter('scene_config_name').value
        self.sim_config_path = self.get_parameter('sim_config_path').value
        self.drone_names = self.get_parameter('drone_names').value
        self.sensor_publish_rate = self.get_parameter('sensor_publish_rate').value
        
        self.get_logger().info(f'Client Address: {self.client_address}')
        self.get_logger().info(f'Scene Config Name: {self.scene_config_name}')
        self.get_logger().info(f'Sim Config Path: {self.sim_config_path}')
        
        # Initialize airsim objects
        self.client = None
        self.world = None
        self.rover = None
        
        # Dictionary to store drone objects and their publishers
        self.drones = {}  # drone_name -> Drone object
        self.imu_publishers = {}  # drone_name -> IMU publisher
        self.image_publishers = {}  # drone_name -> Image publisher
        self.sensor_timers = {}  # drone_name -> timer object
        
        # CV Bridge for image conversion
        self.cv_bridge = CvBridge()
        
        # Run the async initialization (blocking)
        asyncio.run(self.initialize_airsim())
        
        # Start asyncio event loop in separate thread for rover trajectory
        self.loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.async_thread.start()
        
        # Start rover trajectory as background task in the asyncio thread
        self.rover_task = asyncio.run_coroutine_threadsafe(execute_rover_traj1(self.rover), self.loop)
    
    async def initialize_airsim(self):
        # Create a Project AirSim client
        self.client = ProjectAirSimClient(address=self.client_address)

        try:
            # Connect to simulation environment
            self.client.connect()
            self.get_logger().info('Connected to AirSim')

            # Create world and rover objects
            self.world = World(self.client, 
                          scene_config_name=self.scene_config_name, 
                          delay_after_load_sec=2,
                          sim_config_path=self.sim_config_path)
            
            self.rover = Rover(self.client, self.world, "Rover")
            self.rover.enable_api_control()
            self.rover.arm()

            self.get_logger().info('Done setting up AirSim wrapper node and environment.')
            
            # Initialize drones and their ROS2 interfaces
            self.initialize_ros_interface_for_drones()
            
        # logs exception on the console
        except Exception as err:
            self.get_logger().error(f"Exception occurred: {err}")
            projectairsim_log().error(f"Exception occurred: {err}", exc_info=True)

    def initialize_ros_interface_for_drones(self):
        """Initialize ROS2 interface for drones in the simulation"""
        for drone_name in self.drone_names:
            self.get_logger().info(f'Initializing ROS2 interface for drone: {drone_name}')
            
            # Create drone object
            drone = Drone(self.client, self.world, drone_name)
            self.drones[drone_name] = drone
            
            # Create IMU publisher for this drone
            imu_topic = f'{drone_name}/imu'
            self.imu_publishers[drone_name] = self.create_publisher(
                Imu, 
                imu_topic, 
                10
            )
            self.get_logger().info(f'Created IMU publisher: {imu_topic}')
            
            # Create Image publisher for this drone
            image_topic = f'{drone_name}/camera/image_raw'
            self.image_publishers[drone_name] = self.create_publisher(
                Image, 
                image_topic, 
                10
            )
            self.get_logger().info(f'Created Image publisher: {image_topic}')
            
            # Enable API control for the drone
            drone.enable_api_control()
            drone.arm()
            
            # Create timer for publishing sensor data for this drone
            timer_period = 1.0 / self.sensor_publish_rate  # seconds
            timer = self.create_timer(
                timer_period, 
                lambda name=drone_name: self.publish_sensor_data(name)
            )
            self.sensor_timers[drone_name] = timer
            self.get_logger().info(f'Created sensor timer for {drone_name} at {self.sensor_publish_rate} Hz')
    
    def _run_async_loop(self):
        """Run asyncio event loop in separate thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def publish_sensor_data(self, drone_name: str) -> None:
        """Publish IMU and Image data for a specific drone"""
        try:
            drone: Drone = self.drones[drone_name]
            
            # Publish IMU data
            imu_data: Dict = drone.get_imu_data('Imu')
            imu_msg = self.convert_to_imu_msg(imu_data, drone_name)
            self.imu_publishers[drone_name].publish(imu_msg)
            
            # Publish Image data
            image_data = drone.get_images('MainCamera', [ImageType.SCENE])
            if image_data is not None:
                image = unpack_image(list(image_data.values())[0])  # Get the first image
                image_msg = self.convert_to_image_msg(image, drone_name)
                self.image_publishers[drone_name].publish(image_msg)
                
        except Exception as e:
            self.get_logger().error(f'Error publishing sensor data for {drone_name}: {e}')
    
    def convert_to_imu_msg(self, imu_data: Dict, drone_name: str):
        """Convert AirSim IMU data to ROS2 Imu message"""
        imu_msg = Imu()
        
        # Header
        imu_msg.header = Header()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = f'{drone_name}/imu_link'
        
        # Orientation (quaternion)
        imu_msg.orientation.x = imu_data['orientation']['x']
        imu_msg.orientation.y = imu_data['orientation']['y']
        imu_msg.orientation.z = imu_data['orientation']['z']
        imu_msg.orientation.w = imu_data['orientation']['w']
        
        # Angular velocity
        imu_msg.angular_velocity.x = imu_data['angular_velocity']['x']
        imu_msg.angular_velocity.y = imu_data['angular_velocity']['y']
        imu_msg.angular_velocity.z = imu_data['angular_velocity']['z']
        
        # Linear acceleration
        imu_msg.linear_acceleration.x = imu_data['linear_acceleration']['x']
        imu_msg.linear_acceleration.y = imu_data['linear_acceleration']['y']
        imu_msg.linear_acceleration.z = imu_data['linear_acceleration']['z']
        
        return imu_msg
    
    def convert_to_image_msg(self, image_data, drone_name):
        """Convert AirSim image data to ROS2 Image message"""
        # Assuming image_data is a numpy array from AirSim
        image_msg = self.cv_bridge.cv2_to_imgmsg(image_data, encoding="bgr8")
        
        # Update header
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = f'{drone_name}/camera_link'
        
        return image_msg
    
    def shutdown(self):
        """Clean shutdown method"""
        self.rover_task.cancel()
        
        # Stop asyncio event loop gracefully
        if self.loop is not None and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        # Wait for thread to finish
        if self.async_thread is not None and self.async_thread.is_alive():
            self.async_thread.join(timeout=2.0)
        
        if self.client is not None:
            self.get_logger().info('Disconnecting from AirSim')
            self.client.disconnect()

def main(args=None):
    rclpy.init(args=args)
    
    node = AirSimWrapperNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()

if __name__ == '__main__':
    main()