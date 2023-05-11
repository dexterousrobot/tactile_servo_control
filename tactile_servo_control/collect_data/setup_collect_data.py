import os

from tactile_data.utils import save_json_obj


def setup_sensor_image_params(robot, sensor, save_dir=None):

    bbox_dict = {
        'mini': (320-160,    240-160+25, 320+160,    240+160+25),
        'midi': (320-220+10, 240-220-20, 320+220+10, 240+220-20)
    }
    sensor_type = 'midi'  # TODO: Fix hardcoded sensor type

    if 'sim' in robot:
        sensor_image_params = {
            "type": "standard_tactip",
            "image_size": (256, 256),
            "show_tactile": True
        }

    else:
        sensor_image_params = {
            'type': sensor_type,
            'source': 0,
            'exposure': -7,
            'gray': True,
            'bbox': bbox_dict[sensor_type]
        }

    if save_dir:
        save_json_obj(sensor_image_params, os.path.join(save_dir, 'sensor_image_params'))

    return sensor_image_params


def setup_collect_params(robot, task, save_dir=None):

    if robot.split('_')[0] == 'sim':
        robot = 'sim'

    pose_lims_dict = {
        'surface_3d': [(0, 0, 1, -25, -25, 0), (0, 0, 5, 25, 25, 0)],
    }

    shear_lims_dict = {
        'cr':      [(-5, -5, 0, 0, 0, -5), (5, 5, 0, 0, 0, 5)],
        'mg400':   [(-5, -5, 0, 0, 0, -5), (5, 5, 0, 0, 0, 5)],
        'sim':     [(0, 0, 0, 0, 0, 0),    (0, 0, 0, 0, 0, 0)],
    }

    object_poses_dict = {
        'surface_3d': {'surface': (-50, 0, 0, 0, 0, 0)},
        'edge_2d':    {'edge': (0, 0, 0, 0, 0, 0)},
        'edge_3d':    {'edge': (0, 0, 0, 0, 0, 0)},
        'edge_5d':    {'edge': (0, 0, 0, 0, 0, 0)},
    }

    collect_params = {
        'pose_llims': pose_lims_dict[task][0],
        'pose_ulims': pose_lims_dict[task][1],
        'shear_llims': shear_lims_dict[robot][0],
        'shear_ulims': shear_lims_dict[robot][1],
        'object_poses': object_poses_dict[task],
        'sample_disk': True,
        'sort': False,
        'seed': 0
    }

    if robot == 'sim':
        collect_params['sort'] = True

    if save_dir:
        save_json_obj(collect_params, os.path.join(save_dir, 'collect_params'))

    return collect_params


def setup_env_params(robot, save_dir=None):

    if robot.split('_')[0] == 'sim':
        robot = 'sim'

    work_frame_dict = {
        'cr':    (20, -475, 100, -180, 0, 90),
        'mg400': (285,  0, 0, -180, 0, 0),
        'sim':   (650, 0, 50, -180, 0, 0),
    }

    tcp_pose_dict = {
        'cr':    (0, 0, -70, 0, 0, 0),
        'mg400': (0, 0, -50, 0, 0, 0),
        'sim':   (0, 0, -85, 0, 0, 0),
    }  # SHOULD BE ROBOT + SENSOR

    env_params = {
        'robot': robot,
        'stim_name': 'square',
        'speed': 50,
        'work_frame': work_frame_dict[robot],
        'tcp_pose': tcp_pose_dict[robot],
    }

    if 'sim' in robot:
        env_params['speed'] = float('inf')
        env_params['stim_pose'] = (600, 0, 12.5, 0, 0, 0)

    if save_dir:
        save_json_obj(env_params, os.path.join(save_dir, 'env_params'))

    return env_params


def setup_collect_data(robot, sensor, task, save_dir=None):
    collect_params = setup_collect_params(robot, task, save_dir)
    sensor_image_params = setup_sensor_image_params(robot, sensor, save_dir)
    env_params = setup_env_params(robot, save_dir)

    return collect_params, env_params, sensor_image_params


if __name__ == '__main__':
    pass
