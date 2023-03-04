# -*- coding: utf-8 -*-
import os
import pandas as pd

from tactile_learning.utils.utils_learning import save_json_obj


def setup_sensor_params(robot, task, save_dir=None):

    if robot == 'Sim':
        sensor_params = {
            "name": "tactip",
            "type": "standard",
            "core": "no_core",
            "dynamics": {},
            "image_size": (128, 128),
            "turn_off_border": False,
        }

    else:
        sensor_params = {
            'type': 'midi',
            'source': 0,
            'exposure': -7,
            'gray': True,
        }

        bbox_params = {
            'mini': (320-160, 240-160+25, 320+160, 240+160+25),
            'midi': (320-220+10, 240-220-20, 320+220+10, 240+220-20)
        }

        sensor_params['bbox'] = bbox_params[sensor_params['type']]

    if save_dir:
        save_json_obj(sensor_params, os.path.join(save_dir, 'sensor_params'))

    return sensor_params


def setup_pose_params(robot, task, save_dir=None):

    pose_lims_df = pd.DataFrame(
        columns = ['task',       'pose_llims',                      'pose_ulims'],
        data = [
                  ['surface_3d', ( 0, 0,  0.5, -45/2, -45/2, 0),    (0, 0, 5.5, 45/2, 45/2, 0)],
                  ['edge_2d',    (-5, 0, -3.5, 0, 0, -180),         (5, 0, -2.5, 0, 0, 180)],
                  ['edge_3d',    (-5, 0, -3.5, -45/2, -45/2, -180), (5, 0, -2.5, 45/2, 45/2, 180)],
                  ['edge_5d',    (-5, 0, -5.5, -45/2, -45/2, -180), (5, 0, -0.5, 45/2, 45/2, 180)],
        ]
    )
    query_str = f"task=='{task}'"
    pose_params = {
        'pose_llims': pose_lims_df.query(query_str)['pose_llims'].iloc[0],
        'pose_ulims': pose_lims_df.query(query_str)['pose_ulims'].iloc[0],
    }   

    # only do shear move on real robots
    if robot != 'Sim':        
        move_lims_df = pd.DataFrame(
            columns = ['task',       'move_llims',             'move_ulims'],
            data = [
                    ['edge_2d',    (-5, -5, 0, 0, 0, -5),    (5, 5, 0, 0, 0, 5)],
                    ['edge_5d',    (-5, -5, 0, -5, -5, -5),  (5, 5, 0, 5, 5, 5)],
                    ['surface_3d', (-5, -5, 0, -5, -5, -5),  (5, 5, 0, 5, 5, 5)],
            ]
        )
        query_str = f"task=='{task}'"
        pose_params['move_llims'] = move_lims_df.query(query_str)['move_llims'].iloc[0],
        pose_params['move_ulims'] = move_lims_df.query(query_str)['move_ulims'].iloc[0],

    if save_dir:
        save_json_obj(pose_params, os.path.join(save_dir, 'pose_params'))

    return pose_params


def setup_env_params(robot, task, save_dir=None):

    env_params = {
        'robot': robot,
        'stim_name': 'square'
    }

    if robot == 'Sim':
        env_params['stim_pose'] = (600, 0, 0, 0, 0, 0)
        env_params['show_tactile'] = True
        env_params['show_gui'] = True
        env_params['quick_mode'] = False

    else:
        env_params['speed'] = 50 
        # env_params['servo_delay'] = 0.1 

    params_df = pd.DataFrame(
        columns = ['robot', 'task',            'work_frame',           'tcp_pose'],
        data = [
                  ['CR',    'edge',    (20, -475, 69, -180, 0, 0),  (0, 0, -100, 0, 0, 0)],
                  ['CR',    'surface', (10, -425, 69, -180, 0, 0),  (0, 0, -100, 0, 0, 0)],
                  ['MG400', 'edge',    (285,  0, -143, -180, 0, 0), (0, 0, -50, 0, 0, 0)],
                  ['MG400', 'surface', (285,  0, -143, -180, 0, 0), (0, 0, -50, 0, 0, 0)],
                  ['Sim',   'edge',    (650, 0, 40, -180, 0, 0),    (0, 0, 0, 0, 0, 0)],
                  ['Sim',   'surface', (600, 0, 40, -180, 0, 0),    (0, 0, 0, 0, 0, 0)],
        ]
    )
    query_str = f"robot=='{env_params['robot']}' & task=='{task[:-3]}'"
    env_params['work_frame'] = params_df.query(query_str)['work_frame'].iloc[0]
    env_params['tcp_pose'] = params_df.query(query_str)['tcp_pose'].iloc[0]
        
    if save_dir:
        save_json_obj(env_params, os.path.join(save_dir, 'env_params'))
    
    return env_params


def setup_collect_data(reality, task, save_dir=None):
    sensor_params = setup_sensor_params(reality, task, save_dir)
    pose_params = setup_pose_params(reality, task, save_dir)
    env_params = setup_env_params(reality, task, save_dir)

    return sensor_params, pose_params, env_params


if __name__ == '__main__':
    pass
