# -*- coding: utf-8 -*-
"""
python evaluate_model.py -r Sim -m simple_cnn -t edge_2d
"""
import os
import pandas as pd
from torch.autograd import Variable
import torch

from tactile_learning.supervised.models import create_model
from tactile_learning.supervised.image_generator import ImageDataGenerator
from tactile_learning.utils.utils_learning import load_json_obj

from utils_learning import PoseEncoder, ErrorPlotter, csv_row_to_label
from tactile_servo_control.collect_data.utils_collect_data import setup_parse
from tactile_servo_control import BASE_DATA_PATH, BASE_MODEL_PATH

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model_version = ''


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
    target_label_names = label_encoder.target_label_names
    acc_df = pd.DataFrame(columns=[*target_label_names, 'overall_acc'])
    err_df = pd.DataFrame(columns=target_label_names)
    pred_df = pd.DataFrame(columns=target_label_names)
    targ_df = pd.DataFrame(columns=target_label_names)

    for _, batch in enumerate(loader):

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
    pass

    input_args = {
        'tasks':  [['edge_2d'],    "['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d']"],
        'models': [['simple_cnn'], "['simple_cnn', 'posenet_cnn', 'nature_cnn', 'resnet', 'vit']"],
        'robot':  ['Sim',           "['Sim', 'MG400', 'CR']"],
        'device': ['cuda',         "['cpu', 'cuda']"],
    }
    tasks, model_types, robot, device = setup_parse(input_args)

    # test the trained networks
    for model_type in model_types:
        for task in tasks:

            val_data_dirs = [
                os.path.join(BASE_DATA_PATH, robot, task, 'val')
            ]

            # set save dir
            save_dir = os.path.join(BASE_MODEL_PATH, robot, task, model_type + model_version)

            # setup parameters
            task_params = load_json_obj(os.path.join(save_dir, 'task_params'))
            model_params = load_json_obj(os.path.join(save_dir, 'model_params'))
            learning_params = load_json_obj(os.path.join(save_dir, 'learning_params'))
            sensor_params = load_json_obj(os.path.join(save_dir, 'sensor_params'))

            # create the label encoder/decoder
            label_encoder = PoseEncoder(**task_params, device=device)
            
            # create plotter of prediction errors
            error_plotter = ErrorPlotter(
                task_params['label_names'],
                save_dir,
                name='evaluation_error_plot.png'
            )
            
            # create the model
            model = create_model(
                in_dim=sensor_params['image_processing']['dims'],
                in_channels=1,
                out_dim=label_encoder.out_dim,
                model_params=model_params,
                device=device
            )
            model.eval()

            val_generator = ImageDataGenerator(
                data_dirs=val_data_dirs,
                csv_row_to_label=csv_row_to_label,
                **sensor_params['image_processing']
            )

            evaluate_model(
                model,
                label_encoder,
                val_generator,
                learning_params,
                error_plotter,
                device=device
            )
