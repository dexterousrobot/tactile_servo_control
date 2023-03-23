import os
import shutil
import numpy as np

from tactile_data.utils_data import load_json_obj, save_json_obj


def setup_learning(save_dir=None):

    learning_params = {
        'seed': 42,
        'batch_size': 16,
        'epochs': 200,
        'lr': 1e-4,
        'lr_factor': 0.5,
        'lr_patience': 10,
        'adam_decay': 1e-6,
        'adam_b1': 0.9,
        'adam_b2': 0.999,
        'shuffle': True,
        'n_cpu': 1,
    }

    image_processing_params = {
        'dims': (128, 128),
        'bbox': None,
        'thresh': None,
        'stdiz': False,
        'normlz': True,
    }

    augmentation_params = {
        'rshift': (0.025, 0.025),
        'rzoom': None,
        'brightlims': None,
        'noise_var': None,
    }

    preproc_params = {
        'image_processing': image_processing_params,
        'augmentation': augmentation_params
    }

    if save_dir:
        save_json_obj(learning_params, os.path.join(save_dir, 'learning_params'))
        save_json_obj(preproc_params, os.path.join(save_dir, 'preproc_params'))

    return learning_params, preproc_params


def setup_model(model_type, save_dir=None):

    model_params = {
        'model_type': model_type
    }

    if model_type == 'simple_cnn':
        model_params['model_kwargs'] = {
                'conv_layers': [32, 32, 32, 32],
                'conv_kernel_sizes': [11, 9, 7, 5],
                'fc_layers': [512, 512],
                'activation': 'relu',
                'dropout': 0.0,
                'apply_batchnorm': True,
        }

    elif model_type == 'posenet_cnn':
        model_params['model_kwargs'] = {
                'conv_layers': [256, 256, 256, 256, 256],
                'conv_kernel_sizes': [3, 3, 3, 3, 3],
                'fc_layers': [64],
                'activation': 'elu',
                'dropout': 0.0,
                'apply_batchnorm': True,
        }

    elif model_type == 'nature_cnn':
        model_params['model_kwargs'] = {
            'fc_layers': [512, 512],
            'dropout': 0.0,
        }

    elif model_type == 'resnet':
        model_params['model_kwargs'] = {
            'layers': [2, 2, 2, 2],
        }

    elif model_type == 'vit':
        model_params['model_kwargs'] = {
            'patch_size': 32,
            'dim': 128,
            'depth': 6,
            'heads': 8,
            'mlp_dim': 512,
            'pool': 'mean',  # for regression
        }

    else:
        raise ValueError(f'Incorrect model_type specified: {model_type}')

    if save_dir:
        save_json_obj(model_params, os.path.join(save_dir, 'model_params'))

    return model_params


def setup_task(task_name, data_dirs, save_dir=None):
    """
    Returns task specific details.
    """

    label_inds_dict = {
        'surface_3d': [   2, 3, 4   ],  # z Rx', 'Ry'
        'edge_2d':    [0,          5],  # x Rz
        'edge_3d':    [0, 2,       5],  # x z Rz
        'edge_5d':    [0, 2, 3, 4, 5],  # x z Rx Ry Rz
    }

    # get data limits from training data
    pose_llims, pose_ulims = [], []
    for data_dir in data_dirs:
        data_task_params = load_json_obj(os.path.join(data_dir, 'task_params'))
        pose_llims.append(data_task_params['pose_llims'])
        pose_ulims.append(data_task_params['pose_ulims'])

    pose_label_names = data_task_params['pose_label_names']

    task_params = {
        'pose_label_names': pose_label_names,
        'target_label_names': [pose_label_names[i] for i in label_inds_dict[task_name]],
        'pose_llims': tuple(np.min(pose_llims, axis=0).astype(float)),
        'pose_ulims': tuple(np.max(pose_ulims, axis=0).astype(float))
    }

    # save parameters
    if save_dir:
        save_json_obj(task_params, os.path.join(save_dir, 'task_params'))

    return task_params


def setup_training(model_type, task, data_dirs, save_dir=None):
    learning_params, preproc_params = setup_learning(save_dir)
    model_params = setup_model(model_type, save_dir)
    task_params = setup_task(task, data_dirs, save_dir)

    # retain data parameters
    if save_dir:
        shutil.copy(os.path.join(data_dirs[0], 'env_params.json'), save_dir)
        shutil.copy(os.path.join(data_dirs[0], 'sensor_params.json'), save_dir)

    return learning_params, model_params, preproc_params, task_params
