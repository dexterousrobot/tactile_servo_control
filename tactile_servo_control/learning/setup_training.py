import os
import glob
import shutil
import numpy as np

from tactile_data.utils import load_json_obj, save_json_obj
from tactile_data.collect_data.setup_targets import POSE_LABEL_NAMES, SHEAR_LABEL_NAMES


def csv_row_to_label(row):
    row_dict = {label: np.array(row[label]) for label in [*POSE_LABEL_NAMES, *SHEAR_LABEL_NAMES]}
    return row_dict


def setup_learning(save_dir=None):

    # learning_params = {
    #     'seed': 42,
    #     'batch_size': 16,
    #     'epochs': 10,
    #     'lr': 1e-5,
    #     'lr_factor': 0.5,
    #     'lr_patience': 10,
    #     'adam_decay': 1e-6,
    #     'adam_b1': 0.9,
    #     'adam_b2': 0.999,
    #     'shuffle': True,
    #     'n_cpu': 1,
    #     'n_train_batches_per_epoch': None,
    #     'n_val_batches_per_epoch': None,
    # }

    learning_params = {
        "seed": 42, 
        "batch_size": 16, 
        "epochs": 50, 
        "cyclic_base_lr": 1e-08, 
        "cyclic_max_lr": 0.0001, 
        "cyclic_half_period": 5, 
        "cyclic_mode": "triangular", 
        "shuffle": True, 
        "n_cpu": 1, 
        "n_train_batches_per_epoch": None, 
        "n_val_batches_per_epoch": None
    }

    if save_dir:
        save_json_obj(learning_params, os.path.join(save_dir, 'learning_params'))

    return learning_params


def setup_model_image(save_dir=None):

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

    model_image_params = {
        'image_processing': image_processing_params,
        'augmentation': augmentation_params
    }

    if save_dir:
        save_json_obj(model_image_params, os.path.join(save_dir, 'model_image_params'))

    return model_image_params


def setup_model(model_type, save_dir=None):

    if model_type[-4:] == '_mdn':
        model_params = {
            'model_type': model_type[:-4],
            'mdn_kwargs': {
                'n_mdn_components': 1,
                'model_out_dim': 128,
                'hidden_dims': [256, 256],
                'activation': 'relu',
                'noise_type': 'diagonal',
                'fixed_noise_level': None
            }
        }

    else:
        model_params = {
            'model_type': model_type
        }

    if model_params['model_type'] == 'simple_cnn':
        model_params['model_kwargs'] = {
            'conv_layers': [32, 32, 32, 32],
            'conv_kernel_sizes': [11, 9, 7, 5],
            'fc_layers': [512, 512],
            'activation': 'relu',
            'dropout': 0.0,
            'apply_batchnorm': True,
        }

    elif model_params['model_type'] == 'posenet_cnn':
        model_params['model_kwargs'] = {
            'conv_layers': [256, 256, 256, 256, 256],
            'conv_kernel_sizes': [3, 3, 3, 3, 3],
            'fc_layers': [64],
            'activation': 'elu',
            'dropout': 0.0,
            'apply_batchnorm': True,
        }

    elif model_params['model_type'] == 'nature_cnn':
        model_params['model_kwargs'] = {
            'fc_layers': [512, 512],
            'dropout': 0.0,
        }

    elif model_params['model_type'] == 'resnet':
        model_params['model_kwargs'] = {
            'layers': [2, 2, 2, 2]
        }

    elif model_params['model_type'] == 'vit':
        model_params['model_kwargs'] = {
            'patch_size': 32,
            'dim': 128,
            'depth': 6,
            'heads': 8,
            'mlp_dim': 512,
            'pool': 'mean',  # for regression
        }

    elif model_params['model_type'] == 'cnn_mdn_jl':
        model_params['model_kwargs'] = {
            "conv_filters": [16, 32, 64, 128], 
            "conv_kernel_sizes": [11, 9, 7, 5], 
            "conv_padding": "same", 
            "conv_batch_norm": True, 
            "conv_activation": "elu", 
            "conv_pool_size": 2, 
            "fc_units": [512, 512], 
            "fc_activation": "elu", 
            "fc_dropout": 0.1, 
            "mix_components": 1, 
            "pi_dropout": 0.1, 
            "mu_dropout": [0.1, 0.1, 0.2, 0.0, 0.0, 0.1], 
            "sigma_inv_dropout": [0.1, 0.1, 0.2, 0.0, 0.0, 0.1], 
            "mu_min": [-np.inf,]*6, 
            "mu_max": [np.inf,]*6, 
            "sigma_inv_min": [1e-06,]*6, 
            "sigma_inv_max": [1e6,]*6
        }

    else:
        raise ValueError(f'Incorrect model_type specified: {model_type}')

    if save_dir:
        save_json_obj(model_params, os.path.join(save_dir, 'model_params'))

    return model_params


def setup_model_labels(task_name, data_dirs, save_dir=None):
    """
    Returns settings for model labelling of outputs
    """

    target_label_names_dict = {
        'surface_3d':      ['pose_z', 'pose_Rx', 'pose_Ry', 'shear_x', 'shear_y', 'shear_Rz'],
        'surface_3d_pose': ['pose_z', 'pose_Rx', 'pose_Ry'],
        'surface_3d_shear': ['shear_x', 'shear_y', 'shear_Rz'],
    }

    target_weights_dict = {
        'surface_3d':      [1, 1, 1, 1, 1, 1],
        'surface_3d_pose': [1, 1, 1],
        'surface_3d_shear': [1, 1, 2],
    }

    # get data limits from training data
    llims, ulims = [], []
    for data_dir in data_dirs:
        collect_params = load_json_obj(os.path.join(data_dir, 'collect_params'))
        llims.append([*collect_params['pose_llims'], *collect_params['shear_llims']])
        ulims.append([*collect_params['pose_ulims'], *collect_params['shear_ulims']])

    model_label_params = {
        'target_label_names': target_label_names_dict[task_name],
        'target_weights': target_weights_dict[task_name],
        'label_names': [*POSE_LABEL_NAMES, *SHEAR_LABEL_NAMES],
        'llims': tuple(np.min(llims, axis=0).astype(float)),
        'ulims': tuple(np.max(ulims, axis=0).astype(float)),
    }

    # save parameters
    if save_dir:
        save_json_obj(model_label_params, os.path.join(save_dir, 'model_label_params'))

    return model_label_params


def setup_training(model_type, task, data_dirs, save_dir=None):
    learning_params = setup_learning(save_dir)
    model_params = setup_model(model_type, save_dir)
    model_label_params = setup_model_labels(task, data_dirs, save_dir)
    model_image_params = setup_model_image(save_dir)

    is_processed = os.path.isdir(os.path.join(data_dirs[0], 'processed_images'))

    # retain data parameters
    if save_dir:
        shutil.copy(os.path.join(data_dirs[0], 'env_params.json'), save_dir)
        if is_processed:
            shutil.copy(os.path.join(data_dirs[0], 'processed_image_params.json'), save_dir)
        else:
            shutil.copy(os.path.join(data_dirs[0], 'sensor_image_params.json'), save_dir)

    return learning_params, model_params, model_label_params, model_image_params


def setup_training_markers(model_type, task, data_dirs, save_dir=None):
    learning_params = setup_learning(save_dir)
    model_image_params = setup_model(model_type, save_dir)
    model_label_params = setup_model_labels(task, data_dirs, save_dir)

    is_processed = os.path.isdir(os.path.join(data_dirs[0], 'processed_images'))

    # retain data parameters
    if save_dir:
        shutil.copy(os.path.join(data_dirs[0], 'env_params.json'), save_dir)
        if is_processed:
            shutil.copy(os.path.join(data_dirs[0], 'processed_image_params.json'), save_dir)
        else:
            shutil.copy(os.path.join(data_dirs[0], 'sensor_image_params.json'), save_dir)

    return learning_params, model_image_params, model_label_params, {}
