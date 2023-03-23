import os

from tactile_data.utils_data import save_json_obj


def setup_sensor_params(robot, sensor, save_dir=None):

    bbox_dict = {
        'mini': (320-160, 240-160+25, 320+160, 240+160+25),
        'midi': (320-220+10, 240-220-20, 320+220+10, 240+220-20)
    }
    sensor_type = 'midi'

    if robot == 'sim':
        sensor_params = {
            "type": "standard_tactip",
            "image_size": (128, 128),
            "show_tactile": True
        }

    else:
        sensor_params = {
            'type': sensor_type,
            'source': 0,
            'exposure': -7,
            'gray': True,
            'bbox': bbox_dict[sensor_type]
        }

    if save_dir:
        save_json_obj(sensor_params, os.path.join(save_dir, 'sensor_params'))

    return sensor_params


def setup_task_params(robot, task, save_dir=None):

    pose_lims_dict = {
        'surface_3d': [ ( 0, 0, 1, -20, -20,    0), (0, 0, 5, 20, 20, 0) ],
        'edge_2d':    [ (-5, 0, 3,   0,   0, -180), (5, 0, 4, 0, 0, 180) ],
        'edge_3d':    [ (-5, 0, 1,   0,   0, -180), (5, 0, 5, 0, 0, 180) ],
        'edge_5d':    [ (-5, 0, 1, -20, -20, -180),   (5, 0, 5, 20, 20, 180) ],
    }
    
    shear_lims_dict = {
        'cr':      [ (-5, -5, 0, 0, 0, -5), (5, 5, 0, 0, 0, 5) ],
        'mg400':   [ (-5, -5, 0, 0, 0, -5), (5, 5, 0, 0, 0, 5) ],
        'sim':     [ ( 0, 0, 0, 0, 0, 0),   (0, 0, 0, 0, 0, 0) ],
    }

    task_params = {
        'pose_label_names': ["x", "y", "z", "Rx", "Ry", "Rz"],
        'pose_llims': pose_lims_dict[task][0],
        'pose_ulims': pose_lims_dict[task][1],
        'shear_label_names': ["dx", "dy", "dz", "dRx", "dRy", "dRz"],
        'shear_llims': shear_lims_dict[robot][0],
        'shear_ulims': shear_lims_dict[robot][1],
        'sort': False
    }   

    if robot == 'sim':
        task_params['sort'] = True

    if save_dir:
        save_json_obj(task_params, os.path.join(save_dir, 'task_params'))

    return task_params


def setup_env_params(robot, task, save_dir=None):

    work_frame_df = {
        'cr_edge':       [ (20, -475, 100, -180, 0, 90), (0, 0, -70, 0, 0, 0) ],
        'cr_surface':    [ (20, -425, 100, -180, 0, 90), (0, 0, -70, 0, 0, 0) ],
        'mg400_edge':    [ (285,  0, 0, -180, 0, 0),     (0, 0, -50, 0, 0, 0) ],
        'mg400_surface': [ (285,  0, 0, -180, 0, 0),     (0, 0, -50, 0, 0, 0) ],
        'sim_edge':      [ (650, 0, 50, -180, 0, 0),     (0, 0, -85, 0, 0, 0) ],
        'sim_surface':   [ (600, 0, 50, -180, 0, 0),     (0, 0, -85, 0, 0, 0) ],
    }

    env_params = {
        'robot': robot,
        'stim_name': 'square',
        'speed': 50, 
        'work_frame': work_frame_df[robot+'_'+task[:-3]][0],
        'tcp_pose': work_frame_df[robot+'_'+task[:-3]][1]
    }

    if robot == 'sim':
        env_params['stim_pose'] = (600, 0, 0, 0, 0, 0)

    if save_dir:
        save_json_obj(env_params, os.path.join(save_dir, 'env_params'))
    
    return env_params


def setup_collect_data(robot, sensor, task, save_dir=None):
    sensor_params = setup_sensor_params(robot, sensor, save_dir)
    task_params = setup_task_params(robot, task, save_dir)
    env_params = setup_env_params(robot, task, save_dir)

    return sensor_params, task_params, env_params


if __name__ == '__main__':
    pass
