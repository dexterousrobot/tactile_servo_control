"""
python launch_training.py -r cr -s tactip_331 -m simple_cnn -t surface_3d
"""
import os

from tactile_data.tactile_servo_control import BASE_DATA_PATH, BASE_MODEL_PATH
from tactile_data.utils_data import make_dir
from tactile_learning.supervised.image_generator import ImageDataGenerator
from tactile_learning.supervised.models import create_model
from tactile_learning.supervised.train_model_w_metrics import train_model_w_metrics
from tactile_learning.utils.utils_learning import seed_everything
from tactile_servo_control.utils.setup_parse_args import setup_parse_args

from evaluate_model import evaluate_model
from setup_training import setup_training, csv_row_to_label
from utils_learning import LabelEncoder
from utils_plots import RegressErrorPlotter


def launch(
    robot='cr', 
    sensor='tactip_331',
    tasks=['edge_5d'],
    models=['simple_cnn'],
    device='cuda'
):  
    model_version = ''
    task_version = ''  # _pose, _shear

    robot_str, sensor_str, tasks, models, _, device = setup_parse_args(robot, sensor, tasks, models, device)

    for task, model_str in zip(tasks, models):

        # data dirs - list of directories combined in generator
        train_data_dirs = [
            os.path.join(BASE_DATA_PATH, robot_str+'_'+sensor_str, task, 'train'),
        ]
        val_data_dirs = [
            os.path.join(BASE_DATA_PATH, robot_str+'_'+sensor_str, task, 'val'),
        ]

        # setup save dir
        save_dir = os.path.join(BASE_MODEL_PATH, robot_str+'_'+sensor_str, task, model_str + model_version)
        make_dir(save_dir)
        
        # setup parameters
        learning_params, model_params, preproc_params, task_params = setup_training(
            model_str, 
            task,
            train_data_dirs, 
            save_dir
        )  

        # create the label encoder/decoder
        label_encoder = LabelEncoder(task_params, device)

        # create plotter of prediction errors
        error_plotter = RegressErrorPlotter(task_params, save_dir, name='error_plot.png', plot_during_training=True)

        # create the model
        seed_everything(learning_params['seed'])
        model = create_model(
            in_dim=preproc_params['image_processing']['dims'],
            in_channels=1,
            out_dim=label_encoder.out_dim,
            model_params=model_params,
            device=device
        )

        # set generators and loaders
        train_generator = ImageDataGenerator(
            train_data_dirs,
            csv_row_to_label,
            **{**preproc_params['image_processing'], **preproc_params['augmentation']}
        )
        val_generator = ImageDataGenerator(
            val_data_dirs,
            csv_row_to_label,
            **preproc_params['image_processing']
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
        error_plotter.name = 'error_plot_final.png'

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
