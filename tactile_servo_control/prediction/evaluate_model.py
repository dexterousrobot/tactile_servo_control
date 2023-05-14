"""
python evaluate_model.py -r sim -m simple_cnn -t surface_3d
"""
import os
import itertools as it
import pandas as pd
from torch.autograd import Variable
import torch

from tactile_data_shear.tactile_servo_control import BASE_DATA_PATH, BASE_MODEL_PATH
from tactile_image_processing.utils import load_json_obj
from tactile_learning.supervised.models import create_model
from tactile_learning.supervised.image_generator import ImageDataGenerator
from tactile_learning.utils.utils_plots import RegressionPlotter

from tactile_servo_control.learning.setup_training import csv_row_to_label
from tactile_servo_control.utils.label_encoder import LabelEncoder
from tactile_servo_control.utils.parse_args import parse_args


def evaluate_model(
    model,
    label_encoder,
    generator,
    learning_params,
    error_plotter,
    device='cpu'
):

    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    # complete dateframe of predictions and targets
    target_label_names = list(filter(None, label_encoder.target_label_names))
    pred_means_df = pd.DataFrame(columns=target_label_names)
    pred_stdev_df = pd.DataFrame(columns=target_label_names)
    targ_df = pd.DataFrame(columns=target_label_names)

    for _, batch in enumerate(loader):

        # get inputs
        inputs, targ_dict = batch['inputs'], batch['labels']

        # wrap them in a Variable object
        inputs = Variable(inputs).float().to(device)

        # forward pass
        try:
            pred_means, pred_stdev = model.predict(inputs)
        except: # compatibility without MDN
            pred_means = model(inputs)
            pred_stdev = pred_means*0

        # count correct for accuracy metric
        pred_means_dict = label_encoder.decode_label(pred_means)
        pred_stdev_dict = label_encoder.decode_label(pred_stdev)

        # append predictions and labels to dataframes
        batch_pred_means_df = pd.DataFrame.from_dict(pred_means_dict)
        batch_pred_stdev_df = pd.DataFrame.from_dict(pred_stdev_dict)
        batch_targ_df = pd.DataFrame.from_dict(targ_dict)
        pred_means_df = pd.concat([pred_means_df, batch_pred_means_df])
        pred_stdev_df = pd.concat([pred_stdev_df, batch_pred_stdev_df])
        targ_df = pd.concat([targ_df, batch_targ_df])

    # reset indices to be 0 -> test set size
    pred_means_df = pred_means_df.reset_index(drop=True).fillna(0.0)
    pred_stdev_df = pred_stdev_df.reset_index(drop=True).fillna(0.0)
    targ_df = targ_df.reset_index(drop=True).fillna(0.0)

    print("Metrics")
    metrics = label_encoder.calc_metrics(pred_means_df, targ_df)
    err_df, acc_df = metrics['err'], metrics['acc']
    print("evaluated_acc:")
    print(acc_df[[*target_label_names, 'overall_acc']].mean())
    print("evaluated_err:")
    print(err_df[target_label_names].mean())

    # plot full error graph
    metrics['stdev'] = pred_stdev_df
    error_plotter.final_plot(
        pred_means_df, targ_df, metrics
    )


def evaluation(args):

    output_dir = '_'.join([args.robot, args.sensor])

    # test the trained networks
    for args.task, args.model in it.product(args.tasks, args.models):

        model_dir_name = '_'.join(filter(None, [args.model, *args.model_version]))
        task_name = '_'.join(filter(None, [args.task, *args.task_version]))

        val_data_dirs = [
            os.path.join(BASE_DATA_PATH, output_dir, args.task, dir) for dir in args.val_dirs
        ]

        # set model dir
        model_dir = os.path.join(BASE_MODEL_PATH, output_dir, task_name, model_dir_name)

        # setup parameters
        learning_params = load_json_obj(os.path.join(model_dir, 'learning_params'))
        model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
        label_params = load_json_obj(os.path.join(model_dir, 'model_label_params'))
        image_params = load_json_obj(os.path.join(model_dir, 'model_image_params'))

        # configure dataloader
        val_generator = ImageDataGenerator(
            val_data_dirs,
            csv_row_to_label,
            **image_params['image_processing']
        )

        # create the label encoder/decoder and error plotter
        label_encoder = LabelEncoder(label_params, device=args.device)
        error_plotter = RegressionPlotter(label_params, model_dir)

        # create and evaluate the model
        model = create_model(
            in_dim=image_params['image_processing']['dims'],
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
        robot='franka',
        sensor='tactip_1',
        tasks=['surface_3d'],
        task_version=['shear'],
        train_dirs=['train'],
        val_dirs=['val'],
        models=['simple_cnn_mdn_jl'],
        # model_version=['temp'],
        device='cuda'
    )

    evaluation(args)