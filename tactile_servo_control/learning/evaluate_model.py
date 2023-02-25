# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
from torch.autograd import Variable
import torch

from tactile_learning.supervised.models import create_model
from tactile_learning.supervised.image_generator import ImageDataGenerator
from tactile_learning.utils.utils_learning import load_json_obj

from setup_learning import setup_task
from utils_learning import ErrorPlotter, PoseEncoder, csv_row_to_label

from tactile_servo_control import BASE_DATA_PATH, BASE_MODEL_PATH

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def evaluate_model(
    task,
    model,
    label_encoder,
    data_dirs,
    learning_params,
    image_processing_params,
    save_dir,
    error_plotter,
    device='cpu'
):
    # set generators and loaders
    generator = ImageDataGenerator(
        data_dirs=data_dirs,
        csv_row_to_label=csv_row_to_label,
        **image_processing_params
    )

    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    # complete dateframe of predictions and targets
    target_label_names = label_encoder.target_label_names
    acc_df = pd.DataFrame(columns=[*target_label_names, 'overall_acc'])
    err_df = pd.DataFrame(columns=target_label_names)
    pred_df = pd.DataFrame(columns=target_label_names)
    targ_df = pd.DataFrame(columns=target_label_names)

    for i, batch in enumerate(loader):

        # get inputs
        inputs, labels_dict = batch['images'], batch['labels']

        # wrap them in a Variable object
        inputs = Variable(inputs).float().to(device)

        # forward pass
        outputs = model(inputs)

        # count correct for accuracy metric
        predictions_dict = label_encoder.decode_label(outputs)

        # append predictions and labels to dataframes
        batch_pred_df = pd.DataFrame.from_dict(predictions_dict)
        batch_targ_df = pd.DataFrame.from_dict(labels_dict)
        pred_df = pd.concat([pred_df, batch_pred_df])
        targ_df = pd.concat([targ_df, batch_targ_df])

        # get errors and accuracy
        batch_err_df, batch_acc_df = label_encoder.calc_batch_metrics(labels_dict, predictions_dict)

        # append error to dataframe
        err_df = pd.concat([err_df, batch_err_df])
        acc_df = pd.concat([acc_df, batch_acc_df])

    # reset indices to be 0 -> test set size
    pred_df = pred_df.reset_index(drop=True).fillna(0.0)
    targ_df = targ_df.reset_index(drop=True).fillna(0.0)
    acc_df = acc_df.reset_index(drop=True).fillna(0.0)
    err_df = err_df.reset_index(drop=True).fillna(0.0)

    print("Metrics")
    print("evaluated_acc:")
    print(acc_df[[*target_label_names, 'overall_acc']].mean())
    print("evaluated_err:")
    print(err_df[target_label_names].mean())

    # plot full error graph
    error_plotter.final_plot(
        pred_df, targ_df, err_df
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose task from ['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d'].",
        default=['edge_2d']
    )
    parser.add_argument(
        '-m', '--models',
        nargs='+',
        help="Choose model from ['simple_cnn', 'nature_cnn', 'resnet', 'vit'].",
        default=['simple_cnn']
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        help="Choose device from ['cpu', 'cuda'].",
        default='cuda'
    )

    # parse arguments
    args = parser.parse_args()
    tasks = args.tasks
    models = args.models
    device = args.device

    # test the trained networks
    for model_type in models:
        for task in tasks:

            # task specific parameters
            out_dim, label_names = setup_task(task)

            # set save dir
            save_dir = os.path.join(BASE_MODEL_PATH, task, model_type)

            # setup parameters
            network_params = load_json_obj(os.path.join(save_dir, 'model_params'))
            learning_params = load_json_obj(os.path.join(save_dir, 'learning_params'))
            image_processing_params = load_json_obj(os.path.join(save_dir, 'image_processing_params'))

            # get the pose limits used for encoding/decoding pose/predictions
            pose_params = load_json_obj(os.path.join(save_dir, 'pose_params'))
            pose_limits = [pose_params['pose_llims'], pose_params['pose_ulims']]

            # create the model
            model = create_model(
                image_processing_params['dims'],
                out_dim,
                network_params,
                saved_model_dir=save_dir,  # loads weights of best_model.pth
                device=device
            )
            model.eval()

            val_data_dirs = [
                os.path.join(BASE_DATA_PATH, task, 'val')
            ]

            # create the encoder/decoder for labels
            label_encoder = PoseEncoder(label_names, pose_limits, device)

            # create plotter for pose error
            error_plotter = ErrorPlotter(
                target_label_names=label_names,
                save_dir=save_dir,
                name='error_plot.png',
                plot_during_training=False
            )

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
