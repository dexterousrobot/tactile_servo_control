import os
import numpy as np

from tactile_sim.utils.setup_pb_utils import connect_pybullet, load_standard_environment
from tactile_sim.utils.setup_pb_utils import load_stim, set_debug_camera
from tactile_sim.utils.setup_pb_utils import simple_pb_loop
from tactile_sim.embodiments import create_embodiment
from tactile_sim.assets.default_rest_poses import rest_poses_dict

from cri.robot import SyncRobot
from cri.controller import SimController, Controller

from tactile_image_processing.simple_sensors import SimSensor, RealSensor, ReplaySensor


def setup_embodiment(
    env_params={},
    sensor_params={}
):
    env_params['stim_path'] = os.path.join(os.path.dirname(__file__), 'stimuli')

    # setup simulated robot
    if 'sim' in env_params['robot']:
        embodiment = setup_pybullet_env(**env_params, **sensor_params)
        robot = SyncRobot(SimController(embodiment.arm))
        sensor = SimSensor(sensor_params, embodiment)
        robot.speed = env_params.get('speed', float('inf'))

    # setup real robot
    else:
        robot = SyncRobot(Controller[env_params['robot']]())
        sensor = RealSensor(sensor_params)
        robot.speed = env_params.get('speed', 10)

    # if replay overwrite sensor
    if sensor_params['type'] == 'replay':
        sensor = ReplaySensor(sensor_params)

    # settings
    robot.controller.servo_delay = env_params.get('servo_delay', 0.0)
    robot.coord_frame = env_params['work_frame']
    robot.tcp = env_params['tcp_pose']

    return robot, sensor


def setup_pybullet_env(
    embodiment_type='tactile_arm',
    arm_type='ur5',
    sensor_type='standard_tactip',
    image_size=(128, 128),
    show_tactile=False,
    stim_name='square',
    stim_path=os.path.dirname(__file__),
    stim_pose=(600, 0, 12.5, 0, 0, 0),
    show_gui=True,
    **kwargs
):

    timestep = 1/240.0

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
    load_stim(pb, stim_name, np.array(stim_pose)/1e3, fixed_base=True)
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

    env_params = {
        'robot': 'sim',
        # 'stim_name': 'square',
        'stim_name': 'static_keyboard',
        'speed': 50,
        'work_frame': (600, 0, 200, 0, 0, 0),
        'tcp_pose': (600, 0, 0, 0, 0, 0),
        'stim_pose': (600, 82.5, 0, 0, 0, 0),
        # 'stim_pose': (600, 0, 12.5, 0, 0, 0),
        'show_gui': True
    }

    sensor_params = {
        "type": "standard_tactip",
        "image_size": (256, 256),
        "show_tactile": False
    }

    robot = setup_embodiment(
        env_params,
        sensor_params
    )

    simple_pb_loop()
