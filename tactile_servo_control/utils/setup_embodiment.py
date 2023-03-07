import os

from cri.robot import SyncRobot
from cri.controller import Controller
from tactile_servo_control.utils.sensors import SimSensor, RealSensor


def setup_embodiment(
    env_params={},
    sensor_params={}
):
  
    if env_params['robot'] == 'Sim':
        env_params['stim_path'] = os.path.join(os.path.dirname(__file__), 'stimuli')
        embodiment = SyncRobot(Controller[env_params['robot']](sensor_params, env_params))
        embodiment.sensor = SimSensor(embodiment, sensor_params)
  
    else:
        embodiment = SyncRobot(Controller[env_params['robot']]())   
        embodiment.sensor = RealSensor(sensor_params)
        embodiment.speed = env_params['speed']

    # settings
    embodiment.coord_frame = env_params['work_frame']
    embodiment.tcp = env_params['tcp_pose']
    
    # turn on servo mode if a delay has been set
    if 'servo_delay' in env_params:
        embodiment.controller.servo_mode = True
        embodiment.controller.servo_delay = env_params['servo_delay']

    return embodiment


if __name__ == '__main__':
    pass
