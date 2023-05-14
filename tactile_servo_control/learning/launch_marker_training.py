"""
python launch_marker_training.py -r abb -s tactip -m fcn -t edge_2d
"""
import os
import itertools as it

from tactile_data.tactile_servo_control import BASE_DATA_PATH, BASE_MODEL_PATH
from tactile_image_processing.utils import make_dir, load_json_obj
from tactile_learning.supervised.marker_generator import MarkerDataGenerator
from tactile_learning.supervised.models import create_model
from tactile_learning.supervised.train_model import train_model
from tactile_learning.utils.utils_learning import seed_everything
from tactile_learning.utils.utils_plots import RegressionPlotter

from tactile_servo_control.learning.setup_training import setup_training_markers, csv_row_to_label
from tactile_servo_control.prediction.evaluate_model import evaluate_model
from tactile_servo_control.utils.label_encoder import LabelEncoder
from tactile_servo_control.utils.parse_args import parse_args


def launch(args):

    output_dir = '_'.join([args.robot, args.sensor])

    for args.task, args.model in it.product(args.tasks, args.models):

        model_dir_name = '_'.join([args.model, *args.model_version])

        # data dirs - list of directories combined in generator
        train_data_dirs = [
            os.path.join(BASE_DATA_PATH, output_dir, args.task, d) for d in args.train_dirs
        ]
        val_data_dirs = [
            os.path.join(BASE_DATA_PATH, output_dir, args.task, d) for d in args.val_dirs
        ]

        # setup save dir
        save_dir = os.path.join(BASE_MODEL_PATH, output_dir, args.task, model_dir_name)
        make_dir(save_dir)

        # setup parameters
        learning_params, model_params, label_params, _ = setup_training_markers(
            args.model,
            args.task,
            train_data_dirs,
            save_dir
        )
        marker_params_file = os.path.join(train_data_dirs[0], 'processed_marker_params')
        marker_params = load_json_obj(marker_params_file)

        # configure dataloaders
        train_generator = MarkerDataGenerator(
            train_data_dirs,
            csv_row_to_label,
            marker_params['num_markers']
        )
        val_generator = MarkerDataGenerator(
            val_data_dirs,
            csv_row_to_label,
            marker_params['num_markers']
        )

        # create the label encoder/decoder and plotter
        label_encoder = LabelEncoder(label_params, args.device)
        error_plotter = RegressionPlotter(label_params, save_dir)#, plot_during_training=False)

        # create the model
        seed_everything(learning_params['seed'])
        model = create_model(
            in_dim=2*marker_params['num_markers'],
            in_channels=1,
            out_dim=label_encoder.out_dim,
            model_params=model_params,
            device=args.device
        )

        train_model(
            prediction_mode='regression',
            model=model,
            label_encoder=label_encoder,
            train_generator=train_generator,
            val_generator=val_generator,
            learning_params=learning_params,
            save_dir=save_dir,
            error_plotter=error_plotter,
            calculate_train_metrics=False,
            device=args.device
        )

        # perform a final evaluation using the last model
        evaluate_model(
            model,
            label_encoder,
            val_generator,
            learning_params,
            error_plotter,
            device=args.device
        )


if __name__ == "__main__":

    args = parse_args(
        robot='abb',
        sensor='tactip',
        tasks=['edge_2d'],
        train_dirs=['train'],
        val_dirs=['val'],
        models=['fcn'],
        model_version=['markers'],
        device='cuda'
    )

    launch(args)
