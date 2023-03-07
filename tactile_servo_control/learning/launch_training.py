# -*- coding: utf-8 -*-
"""
python launch_training.py -r Sim -m simple_cnn -t edge_2d
"""
import os

from setup_learning import setup_model, setup_learning, setup_task
from evaluate_model import evaluate_model
from utils_learning import ErrorPlotter, PoseEncoder, csv_row_to_label

from tactile_learning.supervised.models import create_model
from tactile_learning.utils.utils_learning import seed_everything, make_dir
from tactile_learning.supervised.image_generator import ImageDataGenerator
from tactile_learning.supervised.train_model_w_metrics import train_model_w_metrics

from tactile_servo_control.collect_data.utils_collect_data import setup_parse
from tactile_servo_control import BASE_DATA_PATH, BASE_MODEL_PATH

model_version = ''


def launch():

    input_args = {
        'tasks':  [['edge_2d'],    "['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d']"],
        'models': [['simple_cnn'], "['simple_cnn', 'posenet_cnn', 'nature_cnn', 'resnet', 'vit']"],
        'robot':  ['Sim',           "['Sim', 'MG400', 'CR']"],
        'device': ['cuda',         "['cpu', 'cuda']"],
    }
    tasks, model_types, robot, device = setup_parse(input_args)

    for task in tasks:
        for model_type in model_types:

            # data dirs - can specify list of directories as these are combined in generator
            train_data_dirs = [
                os.path.join(BASE_DATA_PATH, robot, task, 'train')
            ]
            val_data_dirs = [
                os.path.join(BASE_DATA_PATH, robot, task, 'val')
            ]

            # setup save dir
            save_dir = os.path.join(BASE_MODEL_PATH, robot, task, model_type + model_version)
            make_dir(save_dir)
            
            # setup parameters
            task_params = setup_task(task, train_data_dirs, save_dir)  
            model_params = setup_model(model_type, save_dir)
            learning_params, sensor_params = setup_learning(train_data_dirs, save_dir)

            # create the label encoder/decoder
            label_encoder = PoseEncoder(**task_params, device=device)

            # create plotter of prediction errors
            error_plotter = ErrorPlotter(
                task_params['label_names'],
                save_dir,
                name='error_plot.png',
                plot_during_training=False
            )

            # create the model
            seed_everything(learning_params['seed'])
            model = create_model(
                in_dim=sensor_params['image_processing']['dims'],
                in_channels=1,
                out_dim=label_encoder.out_dim,
                model_params=model_params,
                device=device
            )

            # set generators and loaders
            train_generator = ImageDataGenerator(
                data_dirs=train_data_dirs,
                csv_row_to_label=csv_row_to_label,
                **{**sensor_params['image_processing'], **sensor_params['augmentation']}
            )
            val_generator = ImageDataGenerator(
                data_dirs=val_data_dirs,
                csv_row_to_label=csv_row_to_label,
                **sensor_params['image_processing']
            )

            train_model_w_metrics(
                model,
                label_encoder,
                train_generator,
                val_generator,
                learning_params,
                save_dir,
                error_plotter=error_plotter,
                calculate_train_metrics=False,
                device=device
            )

            # perform a final evaluation using the best model
            error_plotter.name = 'eval_error_plot.png'

            evaluate_model(
                model,
                label_encoder,
                val_generator,
                learning_params,
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
