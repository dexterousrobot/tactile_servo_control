import os
import numpy as np

from tactile_data.utils_data import load_json_obj, save_json_obj


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
            'ei_clip': [[-5, 0, -2.5, 0, 0, -45], [5, 0, 2.5, 0, 0, 45]],
            'error': 'lambda y, r: transform_euler(r, y)',  # SE(3) error
            'ref': [0, 2, 3.5, 0, 0, 0]             
        }

    elif task == 'edge_5d':
        control_params = {
            'kp': [0.5, 1, 0.5, 0.5, 0.5, 0.5],                
            'ki': [0.3, 0, 0.3, 0.1, 0.1, 0.1],
            'ei_clip': [[-5, 0, -2.5, -15, -15, -45], [5, 0, 2.5, 15, 15, 45]],
            'error': 'lambda y, r: transform_euler(r, y)',  # SE(3) error
            'ref': [0, -2, 3.5, 0, 0, 0]              
        }

    if save_dir:
        save_json_obj(control_params, os.path.join(save_dir, 'control_params'))

    return control_params


def update_env_params(env_params, object, model_type, save_dir=None):

    wf_offset_dict = {
        'saddle':  (-10, 0, 18.5, 0, 0, 0),
        'default': (0, 0, 3.5, 0, 0, 0)
    }
    wf_offset = np.array(wf_offset_dict.get(object, wf_offset_dict['default']))

    env_params.update({
        'stim_name': object,
        'work_frame': tuple(env_params['work_frame'] - wf_offset),
        'model_type': model_type,
        'speed': 20,
    })

    if save_dir:
        save_json_obj(env_params, os.path.join(save_dir, 'env_params'))

    return env_params


def update_task_params(task_params, object, model_type, save_dir=None):

    task_params.update({
        'num_iterations': 200,
        'utils_plot'
        'show_plot': True,
        'show_slider': False,
        # 'servo_delay': 0.0,
    })

    if save_dir:
        save_json_obj(task_params, os.path.join(save_dir, 'task_params'))

    return task_params


def setup_servo_control(task, object, model_type, model_dir, save_dir=None):
    env_params = load_json_obj(os.path.join(model_dir, 'env_params'))
    task_params = load_json_obj(os.path.join(model_dir, 'task_params'))
    control_params = setup_control_params(task, save_dir)
    env_params = update_env_params(env_params, object, model_type, save_dir)
    task_params = update_task_params(task_params, object, model_type, save_dir)

    return control_params, env_params, task_params


if __name__ == '__main__':
    pass
