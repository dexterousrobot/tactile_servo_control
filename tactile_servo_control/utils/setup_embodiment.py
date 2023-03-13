import os 

from cri.robot import SyncRobot
from cri.controller import SimController, Controller
from cri.sim.utils.sim_utils import setup_pybullet_env
from tactile_servo_control.utils.sensors import SimSensor, RealSensor


def setup_embodiment(
    env_params={},
    sensor_params={}
):
    env_params['stim_path'] = os.path.join(os.path.dirname(__file__), 'stimuli')
 
    if env_params['robot'] == 'Sim':
        embodiment = setup_pybullet_env(**env_params, **sensor_params)        
        robot = SyncRobot(SimController(embodiment.arm))
        sensor = SimSensor(embodiment, sensor_params)
        # robot = SyncRobot(SimController())
        # sensor = SimSensor(robot, sensor_params)
    
    else:
        robot = SyncRobot(Controller[env_params['robot']]())   
        sensor = RealSensor(sensor_params)
        robot.speed = env_params['speed']

    # settings
    robot.coord_frame = env_params['work_frame']
    robot.tcp = env_params['tcp_pose']
    
    # turn on servo mode if a delay has been set
    if 'servo_delay' in env_params:
        robot.controller.servo_mode = True
        robot.controller.servo_delay = env_params['servo_delay']

    return robot, sensor


if __name__ == '__main__':
    pass
