import asyncio
import threading
import numpy as np
import cv2
from typing import List, Dict
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

from projectairsim import ProjectAirSimClient, World, Rover, Drone
from projectairsim.utils import projectairsim_log, unpack_image
from projectairsim.types import ImageType
from projectairsim.image_utils import ImageDisplay
from rover_hardcoded_trajectories import *


width = 640
height = 480
fps = 30

# Initialize GStreamer
Gst.init(None)

class RTSPServer:
    """Embedded RTSP server using GStreamer"""
    def __init__(self, port=8554, mount_point='/live'):
        self.port = port
        self.mount_point = mount_point
        self.server = None
        self.factory = None
        self.mainloop = None
        self.thread = None
        self.media_pipeline = None
        self.media_appsrc = None
        
    def start(self):
        """Start RTSP server in a separate thread"""
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        print(f"RTSP server starting on rtsp://0.0.0.0:{self.port}{self.mount_point}")
        
    def _run_server(self):
        """Run RTSP server main loop"""
        # Create RTSP server
        self.server = GstRtspServer.RTSPServer()
        self.server.set_service(str(self.port))
        
        # Create media factory with custom configure callback
        self.factory = GstRtspServer.RTSPMediaFactory()
        
        # Set launch pipeline - uses appsrc to receive frames
        pipeline_str = (
            f'( appsrc name=source is-live=true block=false format=time '
            f'caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! '
            f'videoconvert ! '
            f'x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast key-int-max={fps} ! '
            f'rtph264pay name=pay0 pt=96 )'
        )
        
        self.factory.set_launch(pipeline_str)
        self.factory.set_shared(True)
        
        # Connect to media-configure signal to get appsrc
        self.factory.connect('media-configure', self.on_media_configure)
        
        # Attach factory to server
        mount_points = self.server.get_mount_points()
        mount_points.add_factory(self.mount_point, self.factory)
        
        # Attach server to default main context
        self.server.attach(None)
        
        print(f"RTSP server running on rtsp://0.0.0.0:{self.port}{self.mount_point}")
        
        # Run GLib main loop
        self.mainloop = GLib.MainLoop()
        self.mainloop.run()
    
    def on_media_configure(self, factory, media):
        """Callback when media is configured - get appsrc element"""
        element = media.get_element()
        self.media_appsrc = element.get_by_name('source')
        global shared_appsrc
        shared_appsrc = self.media_appsrc
        print("Media configured, appsrc ready")
    
    def stop(self):
        """Stop RTSP server"""
        if self.mainloop:
            self.mainloop.quit()
        if self.thread:
            self.thread.join(timeout=2.0)

rtsp_server = None
shared_appsrc = None

class AirSimWrapper:
    def __init__(self, client_address, 
                 scene_config_name,
                 sim_config_path,
                 drone_names,
                 sensor_publish_rate=20.0,
                 rtsp_port=8554,
                 rtsp_mount_point='/live'):
        
        # Configuration parameters
        self.client_address = client_address
        self.scene_config_name = scene_config_name
        self.sim_config_path = sim_config_path
        self.drone_names = drone_names
        self.sensor_publish_rate = sensor_publish_rate
        
        print(f'Client Address: {self.client_address}')
        print(f'Scene Config Name: {self.scene_config_name}')
        print(f'Sim Config Path: {self.sim_config_path}')
        
        # Start embedded RTSP server
        global rtsp_server
        rtsp_server = RTSPServer(port=rtsp_port, mount_point=rtsp_mount_point)
        rtsp_server.start()
        
        # Initialize airsim objects
        self.client = None
        self.world = None
        self.rover = None
        self.image_display = ImageDisplay(with_bounding_box=False)
        
        # Dictionary to store drone objects
        self.drones = {}  # drone_name -> Drone object
        
        # Run the async initialization (blocking)
        asyncio.run(self.initialize_airsim())
        
        # Start asyncio event loop in separate thread for rover trajectory
        self.loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.async_thread.start()
        
        # Start rover trajectory as background task in the asyncio thread
        self.rover_task = asyncio.run_coroutine_threadsafe(execute_rover_traj1(self.rover), self.loop)
        
        # Sensor data polling
        self.running = True
        self.sensor_thread = threading.Thread(target=self._sensor_polling_loop, daemon=True)
        # self.sensor_thread.start()
    
    async def initialize_airsim(self):
        # Create a Project AirSim client
        self.client = ProjectAirSimClient(address=self.client_address)

        try:
            # Connect to simulation environment
            self.client.connect()
            print('Connected to AirSim')

            # Create world and rover objects
            self.world = World(self.client, 
                          scene_config_name=self.scene_config_name, 
                          delay_after_load_sec=2,
                          sim_config_path=self.sim_config_path)
            
            self.rover = Rover(self.client, self.world, "Rover")
            self.rover.enable_api_control()
            self.rover.arm()

            print('Done setting up AirSim wrapper and environment.')
            
            # Initialize drones
            self.initialize_drones()
            
        # logs exception on the console
        except Exception as err:
            print(f"Exception occurred: {err}")
            projectairsim_log().error(f"Exception occurred: {err}", exc_info=True)

    def initialize_drones(self):
        """Initialize drones in the simulation"""
        for drone_name in self.drone_names:
            print(f'Initializing drone: {drone_name}')
            
            # Create drone object
            drone = Drone(self.client, self.world, drone_name)
            self.drones[drone_name] = drone

            self.client.subscribe(
                drone.sensors["MainCamera"]["scene_camera"],
                lambda _, rgb: self.send_to_rtsp_callback(_, rgb)
            )
            
            # Enable API control for the drone
            drone.enable_api_control()
            drone.arm()
            
            print(f'Drone {drone_name} initialized and armed')
    
    def _run_async_loop(self):
        """Run asyncio event loop in separate thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def _sensor_polling_loop(self):
        """Continuously poll sensor data at specified rate"""
        sleep_time = 1.0 / self.sensor_publish_rate
        
        while self.running:
            for drone_name in self.drone_names:
                self.process_sensor_data(drone_name)
            threading.Event().wait(sleep_time)
    
    def process_sensor_data(self, drone_name: str) -> None:
        """Process and print IMU data for a specific drone"""
        try:
            drone: Drone = self.drones[drone_name]
            
            # Get IMU data
            imu_data: Dict = drone.get_imu_data('Imu')
            
            # Process IMU data (print or log)
            print(f"[{drone_name}] IMU Data:")
            print(f"  Orientation: x={imu_data['orientation']['x']:.3f}, "
                  f"y={imu_data['orientation']['y']:.3f}, "
                  f"z={imu_data['orientation']['z']:.3f}, "
                  f"w={imu_data['orientation']['w']:.3f}")
            print(f"  Angular Velocity: x={imu_data['angular_velocity']['x']:.3f}, "
                  f"y={imu_data['angular_velocity']['y']:.3f}, "
                  f"z={imu_data['angular_velocity']['z']:.3f}")
            print(f"  Linear Acceleration: x={imu_data['linear_acceleration']['x']:.3f}, "
                  f"y={imu_data['linear_acceleration']['y']:.3f}, "
                  f"z={imu_data['linear_acceleration']['z']:.3f}")
                
        except Exception as e:
            print(f'Error processing sensor data for {drone_name}: {e}')
    
    def send_to_rtsp_callback(self, _, image_msg: np.ndarray):
        """Callback wrapper that sends image to RTSP server."""
        try:
            # Check if appsrc is available (client connected)
            if shared_appsrc is None:
                # No client connected yet, skip frame
                return
            
            img_np = unpack_image(image_msg)
            img_np = cv2.resize(img_np, (width, height))
            
            # Create GStreamer buffer from numpy array
            frame_data = img_np.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(frame_data), None)
            buf.fill(0, frame_data)
            buf.pts = Gst.CLOCK_TIME_NONE
            buf.duration = Gst.CLOCK_TIME_NONE
            
            # Push buffer to appsrc (non-blocking)
            ret = shared_appsrc.emit('push-buffer', buf)
            if ret != Gst.FlowReturn.OK:
                print(f'Error pushing buffer: {ret}')
            
        except Exception as e:
            print(f'Error writing to RTSP stream: {e}')
    
    def shutdown(self):
        """Clean shutdown method"""
        print("Shutting down...")
        self.running = False
        
        # Cancel rover task
        if hasattr(self, 'rover_task'):
            self.rover_task.cancel()

        # Stop RTSP server
        if rtsp_server:
            rtsp_server.stop()

        # Stop GStreamer pipeline if active
        try:
            if shared_appsrc:
                shared_appsrc.emit('end-of-stream')
        except Exception as e:
            print(f'Error stopping GStreamer pipeline: {e}')
        
        # Stop asyncio event loop gracefully
        if self.loop is not None and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        # Wait for threads to finish
        if self.async_thread is not None and self.async_thread.is_alive():
            self.async_thread.join(timeout=2.0)
        
        if self.sensor_thread is not None and self.sensor_thread.is_alive():
            self.sensor_thread.join(timeout=2.0)
        
        if self.client is not None:
            print('Disconnecting from AirSim')
            self.client.disconnect()
        
        print("Shutdown complete")

def main():
    # Configuration parameters - modify these as needed
    client_address = '172.25.240.1'
    scene_config_name = 'scene_car_following1.jsonc'
    sim_config_path = '/home/robotics/drone_ws/src/p_airsim_wrapper/config/sim_config'
    drone_names = ['Drone']
    sensor_publish_rate = 20.0  # Hz
    
    # RTSP server configuration
    rtsp_port = 8554  # Port for RTSP server
    rtsp_mount_point = '/live'  # Access at rtsp://<ip>:8554/live
    
    wrapper = AirSimWrapper(
        client_address=client_address,
        scene_config_name=scene_config_name,
        sim_config_path=sim_config_path,
        drone_names=drone_names,
        sensor_publish_rate=sensor_publish_rate,
        rtsp_port=rtsp_port,
        rtsp_mount_point=rtsp_mount_point
    )
    
    try:
        print("AirSim Wrapper running. Press Ctrl+C to exit.")
        print(f"RTSP stream available at: rtsp://localhost:{rtsp_port}{rtsp_mount_point}")
        # Keep main thread alive
        while True:
            threading.Event().wait(1.0)
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    finally:
        wrapper.shutdown()

if __name__ == '__main__':
    main()
