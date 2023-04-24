"""
python evaluate_marker_model.py -r abb -m fcn -t edge_2d
"""
import os
import itertools as it

from tactile_data.tactile_servo_control import BASE_DATA_PATH, BASE_MODEL_PATH, BASE_RUNS_PATH
from tactile_data.utils import load_json_obj
from tactile_learning.supervised.models import create_model
from tactile_learning.supervised.marker_generator import MarkerDataGenerator
from tactile_learning.utils.utils_plots import RegressionPlotter

from tactile_servo_control.learning.setup_training import csv_row_to_label
from tactile_servo_control.prediction.evaluate_model import evaluate_model
from tactile_servo_control.utils.label_encoder import LabelEncoder
from tactile_servo_control.utils.parse_args import parse_args


def evaluation(args):

    output_dir = '_'.join([args.robot, args.sensor])

    # test the trained networks
    for args.task, args.model in it.product(args.tasks, args.models):

        model_dir_name = '_'.join(filter(None, [args.model, *args.model_version]))

        val_data_dirs = [
            os.path.join(BASE_DATA_PATH, output_dir, args.task, dir) for dir in args.val_dirs
            # os.path.join(BASE_RUNS_PATH, output_dir, args.task, model_dir_name)
        ]

        # set model dir
        model_dir = os.path.join(BASE_MODEL_PATH, output_dir, args.task, model_dir_name)

        # setup parameters
        learning_params = load_json_obj(os.path.join(model_dir, 'learning_params'))
        model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
        label_params = load_json_obj(os.path.join(model_dir, 'model_label_params'))
        marker_params = load_json_obj(os.path.join(model_dir, 'processed_marker_params'))

        # configure dataloader
        val_generator = MarkerDataGenerator(
            val_data_dirs,
            csv_row_to_label,   
            marker_params['num_markers']
        )

        # create the label encoder/decoder and error plotter
        label_encoder = LabelEncoder(label_params, device=args.device)
        error_plotter = RegressionPlotter(label_params, model_dir)
        # error_plotter = RegressionPlotter(task_params, val_data_dirs[0], name='error_plot_best.png')

        # create and evaluate the model
        model = create_model(
            in_dim=2*marker_params['num_markers'],
            in_channels=1,
            out_dim=label_encoder.out_dim,
            model_params=model_params,
            saved_model_dir=model_dir,
            device=args.device
        )
        model.eval()

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
        val_dirs=['val'],
        models=['fcn'],
        model_version=['markers'],
        device='cuda'
    )

    evaluation(args)