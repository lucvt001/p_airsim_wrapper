import asyncio
import rclpy
from rclpy.node import Node

from projectairsim import ProjectAirSimClient, World, Rover, Drone
from projectairsim.utils import projectairsim_log
from .rover_hardcoded_trajectories import *

class AirSimWrapperNode(Node):
    def __init__(self):
        super().__init__('airsim_wrapper_node')
        
        # Declare ROS2 parameters
        self.declare_parameter('client_address', '172.25.240.1')
        self.declare_parameter('scene_config_name', 'scene_car_following1.jsonc')
        self.declare_parameter('sim_config_path', 'src/p_airsim_wrapper/config/sim_config')
        
        # Get parameter values
        self.client_address = self.get_parameter('client_address').value
        self.scene_config_name = self.get_parameter('scene_config_name').value
        self.sim_config_path = self.get_parameter('sim_config_path').value
        
        self.get_logger().info(f'Client Address: {self.client_address}')
        self.get_logger().info(f'Scene Config Name: {self.scene_config_name}')
        self.get_logger().info(f'Sim Config Path: {self.sim_config_path}')
        
        # Initialize airsim objects
        self.client = None
        self.world = None
        self.drone = None
        self.rover = None
        
        # Run the async initialization
        asyncio.run(self.initialize_airsim())
        asyncio.run(execute_rover_traj1(self.rover))
    
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
            
        # logs exception on the console
        except Exception as err:
            self.get_logger().error(f"Exception occurred: {err}")
            projectairsim_log().error(f"Exception occurred: {err}", exc_info=True)
    
    def shutdown(self):
        """Clean shutdown method"""
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