# -*- coding: utf-8 -*-
import os
import pandas as pd

from tactile_learning.utils.utils_learning import save_json_obj


def setup_sensor_params(reality, task, save_dir=None):

    if reality == 'sim':
        sensor_params = {
            "name": "tactip",
            "type": "standard",
            "core": "no_core",
            "dynamics": {},
            "image_size": [128, 128],
            "turn_off_border": False,
        }

    elif reality == 'real':
        sensor_params = {
            'type': 'midi',
            'source': 0,
            'exposure': -7,
            'gray': True,
        }

        if sensor_params['type'] == 'mini':
            sensor_params['bbox'] = [320-160, 240-160+25, 320+160, 240+160+25] 

        elif sensor_params['type'] == 'midi':
            sensor_params['bbox'] = [320-220+10, 240-220-20, 320+220+10, 240+220-20] 

    if save_dir:
        save_json_obj(sensor_params, os.path.join(save_dir, 'sensor_params'))

    return sensor_params


def setup_pose_params(reality, task, save_dir=None):

    pose_lims_df = pd.DataFrame(
        columns = ['reality', 'task',    'pose_llims',                      'pose_ulims'],
        data = [
                  ['sim',  'surface_3d', ( 0, 0, 0.5, -45/2, -45/2, 0),     (0, 0, 5.5, 45/2, 45/2, 0)],
                  ['sim',  'edge_2d',    (-5, 0, 2.5, 0, 0, -180),          (5, 0, 3.5, 0, 0, 180)],
                  ['sim',  'edge_3d',    (-5, 0, 2.5, -45/2, -45/2, -180),  (5, 0, 3.5, 45/2, 45/2, 180)],
                  ['sim',  'edge_5d',    (-5, 0, 1.5, -45/2, -45/2, -180),  (5, 0, 5.5, 45/2, 45/2, 180)],
                  ['real', 'edge_2d',    (0, -5, -1, 0, 0, -180),           (0, 5, 1, 0, 0, 180)],
        ]
    )
    query_str = f"reality=='{reality}' & task=='{task}'"
    pose_params = {
        'pose_llims': pose_lims_df.query(query_str)['pose_llims'].iloc[0],
        'pose_ulims': pose_lims_df.query(query_str)['pose_ulims'].iloc[0],
    }   

    # only do shear move in real
    if reality == 'real':        
        move_lims_df = pd.DataFrame(
            columns = ['task',       'move_llims',             'move_ulims'],
            data = [
                    ['edge_2d',    (-5, -5, 0, 0, 0, -5),    (5, 5, 0, 0, 0, 5)],
                    ['edge_5d',    (-5, -5, 0, -5, -5, -5),  (5, 5, 0, 5, 5, 5)],
                    ['surface_3d', (-5, -5, 0, -5, -5, -5),  (5, 5, 0, 5, 5, 5)],
            ]
        )
        query_str = f"task=='{task}'"
        pose_params['pose_llims'] = move_lims_df.query(query_str)['move_llims'].iloc[0],
        pose_params['pose_ulims'] = move_lims_df.query(query_str)['move_ulims'].iloc[0],

    if save_dir:
        save_json_obj(pose_params, os.path.join(save_dir, 'pose_params'))

    return pose_params


def setup_env_params(reality, task, save_dir=None):

    if reality == 'real':
        env_params = {
            'robot': 'CR3',
            'tcp_pose': (0, 0, -50, 0, 0, 0),
            'stim_name': 'square',
            'linear_speed': 10, 
            'angular_speed': 10,
        } 

    elif reality == 'sim':
        env_params = {
            'robot': 'UR5',
            'tcp_pose': (0, 0, 50, 0, 0, 0),
            'stim_name': 'square',
            'stim_pose': (600, 0, 0, 0, 0, 0),
            'quick_mode': False,
        }

    work_frame_df = pd.DataFrame(
        columns = ['reality', 'robot', 'task',     'work_frame'],
        data = [
                  ['real',    'CR3',   'edge',     (0, -370,  115, -180, 0, 180)],
                  ['real',    'CR3',   'surface',  (0, -370,  115, -180, 0, 180)],
                  ['real',    'MG400', 'edge',     (285,  0, -143, -180, 0, 0)],
                  ['real',    'MG400', 'surface',  (285,  0, -143, -180, 0, 0)],
                  ['sim',     'UR5',   'edge',     (650,   0,  40, -180, 0, 0)],
                  ['sim',     'UR5',   'surface',  (610, -55,  40, -180, 0, 0)],
        ]
    )
    query_str = f"reality=='{reality}' & robot=='{env_params['robot']}' & task=='{task[:-3]}'"
    env_params['work_frame'] = work_frame_df.query(query_str)['work_frame'].iloc[0]
        
    if save_dir:
        save_json_obj(env_params, os.path.join(save_dir, 'env_params'))
    
    env_params.update({'show_gui': True, 'show_tactile': True})

    return env_params


def setup_collect_data(reality, task, save_dir=None):
    sensor_params = setup_sensor_params(reality, task, save_dir)
    pose_params = setup_pose_params(reality, task, save_dir)
    env_params = setup_env_params(reality, task, save_dir)

    return sensor_params, pose_params, env_params


if __name__ == '__main__':
    pass
