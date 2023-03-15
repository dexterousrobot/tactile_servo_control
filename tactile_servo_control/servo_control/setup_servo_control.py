# -*- coding: utf-8 -*-
import os
import numpy as np

from tactile_learning.utils.utils_learning import load_json_obj, save_json_obj
from cri.transforms import transform_euler


def setup_control_params(task, save_dir=None):

    if task == 'surface_3d':
        control_params = {
            'kp': [1, 1, 0.5, 0.5, 0.5, 1],                 
            'ki': [0, 0, 0.3, 0.1, 0.1, 0],                
            'ei_clip': [[0, 0, 0, -30, -30, 0], [0, 0, 5, 30, 30, 0]],        
            'error': 'lambda y, r: transform_euler(r, y)',  # SE(3) error
            'ref': [0, 1, 3, 0, 0, 0]              
        }

    elif task == 'edge_2d':
        control_params = {
            'kp': [0.5, 1, 0, 0, 0, 0.5],                 
            'ki': [0.3, 0, 0, 0, 0, 0.1],                
            'ei_clip': [[-5, 0, 0, 0, 0, -45], [5, 0, 0, 0, 0,  45]],          
            'error': 'lambda y, r: transform_euler(r, y)',  # SE(3) error
            'ref': [0, 2, 0, 0, 0, 0]              
        }

    elif task == 'edge_3d':
        control_params = {
            'kp': [0.5, 1, 0.5, 0, 0, 0.5],                 
            'ki': [0.3, 0, 0.3, 0, 0, 0.1],                
            'ei_clip': [[0, -5, 0, 0, 0, -45], [0, 5, 5, 0, 0, 45]],
            'error': 'lambda y, r: transform_euler(r, y)',  # SE(3) error
            'ref': [0, 2, 3, 0, 0, 0]             
        }

    elif task == 'edge_5d':
        control_params = {
            'kp': [1, 0.5, 0.5, 0.5, 0.5, 0.5],                
            'ki': [0, 0.3, 0.3, 0.1, 0.1, 0.1],
            'ei_clip': [[0, -5, 0, -30, -30, -45], [0, 5, 5, 30, 30, 45]],
            'error': 'lambda y, r: transform_euler(r, y)',  # SE(3) error
            'ref': [0, 2, 3, 0, 0, 0]              
        }

    if save_dir:
        save_json_obj(control_params, os.path.join(save_dir, 'control_params'))

    # convert error into function handle if exists
    if 'error' in control_params:
        control_params['error'] = eval(control_params['error'])

    return control_params


def setup_env_params(stimulus, data_dir, save_dir=None):

    wf_offset_dict = {
        'saddle':  (-10, 0, 18.5, 0, 0, 0),
        'default': (0, 0, 4.0, 0, 0, 0)
    }
    wf_offset = np.array(wf_offset_dict.get(stimulus, wf_offset_dict['default']))

    env_params = load_json_obj(os.path.join(data_dir, 'env_params'))

    env_params.update({
        'stim_name': stimulus,
        'num_iterations': 200,
        'work_frame': tuple(env_params['work_frame'] - wf_offset),
        'model_type': 'simple_cnn',
        'speed': 20,
        # 'servo_mode': True,
        # 'servo_delay': 0.0,
    })

    if save_dir:
        save_json_obj(env_params, os.path.join(save_dir, 'env_params'))

    return env_params


if __name__ == '__main__':
    pass
