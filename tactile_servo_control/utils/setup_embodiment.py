import os

from cri.robot import SyncRobot, AsyncRobot
from cri.controller import SimController
from cri.controller import CRController
from cri.controller import MG400Controller

from tactile_servo_control.utils.sensors import SimSensor, RealSensor


def setup_embodiment_sim(
    env_params={},
    sensor_params={}
):
    env_params['stim_path'] = os.path.join(os.path.dirname(__file__), 'stimuli')

    # setup the embodiment
    embodiment = SyncRobot(SimController(sensor_params, env_params))   
    embodiment.sensor = SimSensor(embodiment, sensor_params)

    # settings
    embodiment.coord_frame = env_params['work_frame']
    embodiment.tcp = env_params['tcp_pose']

    return embodiment


def setup_embodiment_real(
    env_params={},    
    sensor_params={},
):
    linear_speed = env_params.get('linear_speed', 10)
    angular_speed = env_params.get('angular_speed', 10)

    # setup the embodiment
    if env_params['robot'] == 'MG400':
        embodiment = AsyncRobot(SyncRobot(MG400Controller()))
    elif env_params['robot'] == 'CR3':
        embodiment = AsyncRobot(SyncRobot(CRController()))
    embodiment.sensor = RealSensor(sensor_params)

    # settings
    embodiment.coord_frame = env_params['work_frame']
    embodiment.tcp = env_params['tcp_pose']
    embodiment.linear_speed = linear_speed
    embodiment.angular_speed = angular_speed

    return embodiment


def setup_embodiment(
    reality, 
    env_params, 
    sensor_params
):
    if reality == 'real':
        embodiment = setup_embodiment_real(env_params, sensor_params)
    
    elif reality == 'sim':
        embodiment = setup_embodiment_sim(env_params, sensor_params)

    return embodiment


if __name__ == '__main__':
    pass
