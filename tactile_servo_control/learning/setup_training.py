import os
import glob
import shutil
import numpy as np

from tactile_data.utils import load_json_obj, save_json_obj
from tactile_data.collect_data.setup_targets import POSE_LABEL_NAMES


def csv_row_to_label(row):
    row_dict = {label: np.array(row[label]) for label in POSE_LABEL_NAMES}
    return row_dict


def setup_learning(save_dir=None):

    learning_params = {
        'seed': 42,
        'batch_size': 16,
        'epochs': 2,
        'lr': 1e-4,
        'lr_factor': 0.5,
        'lr_patience': 10,
        'adam_decay': 1e-6,
        'adam_b1': 0.9,
        'adam_b2': 0.999,
        'shuffle': True,
        'n_cpu': 1,
        'n_train_batches_per_epoch': None,
        'n_val_batches_per_epoch': None,
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

    if 'simple_cnn' in model_type:
        model_params = {
            'model_type': 'simple_cnn',
            'model_kwargs': {
                'conv_layers': [32, 32, 32, 32],
                'conv_kernel_sizes': [11, 9, 7, 5],
                'fc_layers': [512, 512],
                'activation': 'relu',
                'dropout': 0.0,
                'apply_batchnorm': True,
            }
        }

    elif 'posenet_cnn' in model_type:
        model_params = {
            'model_type': 'posenet_cnn',
            'model_kwargs': {
                'conv_layers': [256, 256, 256, 256, 256],
                'conv_kernel_sizes': [3, 3, 3, 3, 3],
                'fc_layers': [64],
                'activation': 'elu',
                'dropout': 0.0,
                'apply_batchnorm': True,
            }
        }

    elif 'nature_cnn' in model_type:
        model_params = {
            'model_type': 'nature_cnn',
            'model_kwargs': {
                'fc_layers': [512, 512],
                'dropout': 0.0,
            }
        }
        

    elif 'resnet' in model_type:
        model_params = {
            'model_type': 'resnet',
            'model_kwargs': {
                'layers': [2, 2, 2, 2]
            }
        }

    elif 'vit' in model_type:
        model_params = {
            'model_type': 'vit',
            'model_kwargs': {
                'patch_size': 32,
                'dim': 128,
                'depth': 6,
                'heads': 8,
                'mlp_dim': 512,
                'pool': 'mean',  # for regression
            }
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

    target_label_names_dict = {
        'surface_3d': ['pose_z', 'pose_Rx', 'pose_Ry'],
        'edge_2d':    ['pose_x', 'pose_Rz'],
        'edge_3d':    ['pose_x', 'pose_z', 'pose_Rz'],
        'edge_5d':    ['pose_x', 'pose_z', 'pose_Rx', 'pose_Ry', 'pose_Rz'],
    }

    target_weights_dict = {
        'surface_3d': [1, 1, 1],
        'edge_2d':    [1, 1],
        'edge_3d':    [1, 1, 1],
        'edge_5d':    [1, 1, 1, 1, 1],
    }

    # get data limits from training data
    llims, ulims = [], []
    for data_dir in data_dirs:
        collect_params = load_json_obj(os.path.join(data_dir, 'collect_params'))
        llims.append(collect_params['pose_llims'])
        ulims.append(collect_params['pose_ulims'])

    task_params = {
        'target_label_names': target_label_names_dict[task_name],
        'target_weights': target_weights_dict[task_name],
        'label_names': POSE_LABEL_NAMES,
        'llims': tuple(np.min(llims, axis=0).astype(float)),
        'ulims': tuple(np.max(ulims, axis=0).astype(float)),
        'periodic_label_names': ['pose_Rz']
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
        for file_name in glob.glob(os.path.join(data_dirs[0], '*_params.json')):
            shutil.copy(file_name, save_dir)

        # if there is sensor process params, overwrite
        sensor_proc_params_file = os.path.join(save_dir, 'sensor_process_params.json')
        if os.path.isfile(sensor_proc_params_file):
            os.replace(sensor_proc_params_file, os.path.join(save_dir, 'sensor_params.json'))

    return learning_params, model_params, preproc_params, task_params
