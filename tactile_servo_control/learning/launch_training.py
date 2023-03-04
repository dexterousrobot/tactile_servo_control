# -*- coding: utf-8 -*-
"""
python launch_training.py -t surface_3d edge_2d edge_3d edge_5d
"""
import os
import shutil

from tactile_learning.supervised.models import create_model
from tactile_learning.utils.utils_learning import seed_everything, make_dir
from tactile_learning.supervised.image_generator import ImageDataGenerator
from tactile_learning.supervised.train_model_w_metrics import train_model_w_metrics

from setup_learning import setup_model, setup_learning, setup_task
from utils_learning import ErrorPlotter, PoseEncoder, get_pose_limits, csv_row_to_label
from train_model import train_model
from evaluate_model import evaluate_model

from tactile_servo_control import BASE_DATA_PATH, BASE_MODEL_PATH
from tactile_servo_control.collect_data.utils_collect_data import setup_parse

data_version = ''
model_version = '_test'


def launch():

    input_args = {
        'tasks':   [['edge_2d'],    "['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d']"],
        'models':  [['simple_cnn'], "['simple_cnn', 'posenet_cnn', 'nature_cnn', 'resnet', 'vit']"],
        'reality': ['sim',          "['sim', 'real'"],
        'device':  ['cuda',         "['cpu', 'cuda']"],
    }
    tasks, model_types, reality, device = setup_parse(input_args)

    for task in tasks:
        for model_type in model_types:

            # setup save dir
            save_dir = os.path.join(BASE_MODEL_PATH, reality, task, model_type + model_version)
            make_dir(save_dir)

            # data dir - can specify list of directories as these are combined in generator
            train_data_dirs = [
                os.path.join(BASE_DATA_PATH, reality, task, 'train' + data_version)
            ]
            val_data_dirs = [
                os.path.join(BASE_DATA_PATH, reality, task, 'val' + data_version)
            ]
            
            # setup parameters
            task_params = setup_task(task, save_dir)   #out_dim, label_names
            network_params = setup_model(model_type, save_dir)
            learning_params, image_processing_params, augmentation_params = setup_learning(save_dir)
            pose_limits = get_pose_limits(train_data_dirs, save_dir)

            # keep record of sensor params
            shutil.copy(os.path.join(BASE_DATA_PATH, reality, task, 'train' + data_version, 'sensor_params.json'), save_dir)

            # create the encoder/decoder for labels
            label_encoder = PoseEncoder(task_params['label_names'], pose_limits, device)

            # create instance for plotting errors
            error_plotter = ErrorPlotter(
                task_params['label_names'],
                save_dir,
                name='error_plot.png',
                plot_during_training=False
            )

            # create and train model
            seed_everything(learning_params['seed'])
            model = create_model(
                image_processing_params['dims'],
                task_params['out_dim'],
                network_params,
                device=device
            )

            train_model(
                model,
                label_encoder,
                train_data_dirs,
                val_data_dirs,
                csv_row_to_label,
                learning_params,
                image_processing_params,
                augmentation_params,
                save_dir,
                error_plotter=error_plotter,
                calculate_train_metrics=False,
                device=device
            )

            # perform a final evaluation using the best model
            evaluate_model(
                task,
                model,
                label_encoder,
                val_data_dirs,
                learning_params,
                image_processing_params,
                save_dir,
                error_plotter,
                device=device
            )


if __name__ == "__main__":

    # for profiling and debugging slow functions
    # import cProfile
    # import pstats
    # pstats.Stats(
    #     cProfile.Profile().run("launch()")
    # ).sort_stats(
    #     pstats.SortKey.TIME
    # ).print_stats(20)

    launch()
