import asyncio
from projectairsim import Rover
from projectairsim.utils import projectairsim_log

async def execute_rover_traj1(rover: Rover):
    await (await rover.set_rover_controls(engine=0.2, steering_angle=0, brake=0))
    await asyncio.sleep(5)
    await (await rover.set_rover_controls(engine=0, steering_angle=0, brake=0.4))