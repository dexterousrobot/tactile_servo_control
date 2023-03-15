import os 
import numpy as np

from tactile_sim.utils.pybullet_utils import connect_pybullet, load_standard_environment
from tactile_sim.utils.pybullet_utils import load_stim, set_debug_camera
from tactile_sim.embodiments import create_embodiment
from tactile_sim.assets.default_rest_poses import rest_poses_dict

from cri.robot import SyncRobot
from cri.controller import SimController, Controller

from tactile_servo_control.utils.sensors import SimSensor, RealSensor


def setup_embodiment(
    env_params={},
    sensor_params={}
):
    env_params['stim_path'] = os.path.join(os.path.dirname(__file__), 'stimuli')
 
    # setup simulated robot
    if env_params['robot'] == 'Sim':
        embodiment = setup_pybullet_env(**env_params, **sensor_params)        
        robot = SyncRobot(SimController(embodiment.arm))
        sensor = SimSensor(embodiment, sensor_params)
    
    # setup real robot
    else:
        robot = SyncRobot(Controller[env_params['robot']]())   
        sensor = RealSensor(sensor_params)

    # settings
    robot.speed = env_params.get('speed', 10)
    robot.controller.servo_delay = env_params.get('servo_delay', 0.0)
    robot.coord_frame = env_params['work_frame']
    robot.tcp = env_params['tcp_pose']

    return robot, sensor


def setup_pybullet_env(
    embodiment_type='tactile_arm',
    arm_type='ur5',
    sensor_type='standard_tactip',
    image_size=(128,128),
    show_tactile=False,
    stim_name='square',
    stim_path=os.path.dirname(__file__),        
    stim_pose=(600, 0, 0, 0, 0, 0),
    **kwargs
):
    
    timestep = 1/240.0
    show_gui = True

    # define sensor parameters
    robot_arm_params = {
        "type": arm_type,
        "rest_poses": rest_poses_dict[arm_type],
        "tcp_lims": np.column_stack([-np.inf*np.ones(6), np.inf*np.ones(6)]),
    }

    tactile_sensor_params = {
        "type": sensor_type,
        "core": "no_core",
        "dynamics": {},  # {'stiffness': 50, 'damping': 100, 'friction': 10.0},
        "image_size": image_size,
        "turn_off_border": False,
        "show_tactile": show_tactile,
    }

    # set debug camera position
    visual_sensor_params = {
        'image_size': [128, 128],
        'dist': 0.25,
        'yaw': 90.0,
        'pitch': -25.0,
        'pos': [0.6, 0.0, 0.0525],
        'fov': 75.0,
        'near_val': 0.1,
        'far_val': 100.0,
        'show_vision': False
    }

    pb = connect_pybullet(timestep, show_gui)
    load_standard_environment(pb)
    stim_name = os.path.join(stim_path, stim_name, stim_name+'.urdf')
    load_stim(pb, stim_name, np.array(stim_pose)/1e3)
    embodiment = create_embodiment(
        pb,
        embodiment_type,
        robot_arm_params,
        tactile_sensor_params,
        visual_sensor_params
    )
    set_debug_camera(pb, visual_sensor_params)
    return embodiment


if __name__ == '__main__':
    pass
