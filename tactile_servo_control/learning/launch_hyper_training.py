"""
python launch_training.py -r sim -s tactip -m simple_cnn -t surface_3d
"""
import os
import copy
import shutil
import itertools as it
import pandas as pd

from functools import partial
from hyperopt import tpe, hp, fmin, Trials, STATUS_OK, STATUS_FAIL

from tactile_data.tactile_servo_control import BASE_DATA_PATH, BASE_MODEL_PATH
from tactile_image_processing.utils import make_dir, save_json_obj
from tactile_learning.supervised.image_generator import ImageDataGenerator
from tactile_learning.supervised.models import create_model
from tactile_learning.supervised.train_model import train_model
from tactile_learning.utils.utils_learning import seed_everything
from tactile_learning.utils.utils_plots import RegressionPlotter

from tactile_servo_control.learning.setup_training import setup_training, csv_row_to_label
from tactile_servo_control.prediction.evaluate_model import evaluate_model
from tactile_servo_control.utils.label_encoder import LabelEncoder
from tactile_servo_control.utils.parse_args import parse_args


# build hyperopt objective function
def create_objective_func(
            train_generator,
            val_generator,
            learning_params,
            model_params,
            image_params,
            label_params,
            save_dir,
            error_plotter=None,
            device='cpu'
        ):
    trial = 0
    lowest_val_loss = float('inf')

    def objective_func(args):
        nonlocal trial, lowest_val_loss

        params_list = [
            learning_params,
            label_params,
            model_params['model_kwargs'],
        ]

        # set parameters: indexed and non-indexed
        print(f"\nTrial: {trial+1}\n")
        for arg, val in args.items():
            print(f'{arg}:{val}')

            index = arg.split('_')[-1]
            for params in params_list:
                if index.isdigit():
                    arg_0 = arg.replace('_'+index, '')
                    if arg_0 in params:
                        params[arg_0][int(index)] = val
                else:
                    if arg in params:
                        params[arg] = val

        # create labels and model
        label_encoder = LabelEncoder(label_params, device)

        seed_everything(learning_params['seed'])
        model = create_model(
            in_dim=image_params['image_processing']['dims'],
            in_channels=1,
            out_dim=label_encoder.out_dim,
            model_params=model_params,
            display_model=False,
            device=device
        )

        try:
            val_loss, train_time = train_model(
                'regression',
                model,
                label_encoder,
                train_generator,
                val_generator,
                learning_params,
                save_dir,
                device=device
            )

            results = {
                "loss": val_loss,
                "status": STATUS_OK,
                "training_time": train_time,
            }
            print(f"Loss: {results['loss']:.2}\n")

        except:
            results = {
                "loss": None,
                "status": STATUS_FAIL,
                "training_time": None,
            }
            print("Aborted trial: Resource exhausted error\n")

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss

            model_file = os.path.join(save_dir, 'best_model.pth')
            shutil.copyfile(model_file, os.path.join(save_dir, 'hyper_best_model.pth'))

            save_json_obj(model_params, os.path.join(save_dir, 'model_params'))
            save_json_obj(learning_params, os.path.join(save_dir, 'learning_params'))
            save_json_obj(label_params, os.path.join(save_dir, 'model_label_params'))

            if error_plotter:
                error_plotter.block = False
                evaluate_model(
                    model,
                    label_encoder,
                    val_generator,
                    learning_params,
                    error_plotter,
                    device
                )

        trial += 1
        return {**results, 'trial': trial}

    return objective_func


def make_trials_df(trials):
    trials_df = pd.DataFrame()
    for i, trial in enumerate(trials):
        trial_params = {k: v[0] if len(v) > 0 else None for k, v in trial['misc']['vals'].items()}
        trial_row = pd.DataFrame(format_params(trial_params), index=[i])
        trial_row['loss'] = trial['result']['loss']
        trial_row['status'] = trial['result']['status']
        trial_row['training_time'] = trial['result']['training_time']
        trials_df = pd.concat([trials_df, trial_row])
    return trials_df


def format_params(params):
    params_conv = copy.deepcopy(params)
    if 'activation' in params_conv:
        params_conv['activation'] = ('relu', 'elu')[params['activation']]
    if 'conv_layers' in params_conv:
        params_conv['conv_layers'] = ('[16,]*4', '[32,]*4')[params['conv_layers']]
    return params_conv


def launch(args, space, max_evals=20, n_startup_jobs=10):

    output_dir = '_'.join([args.robot, args.sensor])

    for args.task, args.model in it.product(args.tasks, args.models):

        model_dir_name = '_'.join(filter(None, [args.model, *args.model_version]))

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
        learning_params, model_params, label_params, image_params = setup_training(
            args.model,
            args.task,
            train_data_dirs,
            save_dir
        )

        # set generators
        train_generator = ImageDataGenerator(
            train_data_dirs,
            csv_row_to_label,
            **{**image_params['image_processing'], **image_params['augmentation']}
        )
        val_generator = ImageDataGenerator(
            val_data_dirs,
            csv_row_to_label,
            **image_params['image_processing']
        )

        # create the error plotter
        error_plotter = RegressionPlotter(label_params, save_dir)

        # create the hyperparameter optimization
        trials = Trials()
        obj_func = create_objective_func(
            train_generator,
            val_generator,
            learning_params,
            model_params,
            image_params,
            label_params,
            save_dir,
            error_plotter,
            args.device
        )

        # optimize the model
        opt_params = fmin(
            obj_func,
            space,
            max_evals=max_evals,
            trials=trials,
            algo=partial(tpe.suggest, n_startup_jobs=n_startup_jobs)
        )

        # finish and report
        print(opt_params)
        trials_df = make_trials_df(trials)
        trials_df.to_csv(os.path.join(save_dir, "trials.csv"), index=False)

        # keep best model
        best_model_file = os.path.join(save_dir, 'hyper_best_model.pth')
        shutil.copy(best_model_file, os.path.join(save_dir, 'best_model.pth'))


if __name__ == "__main__":

    args = parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['edge_2d'],
        train_dirs=['train_data'],
        val_dirs=['val_data'],
        models=['simple_cnn'],
        model_version=['hyp_temp'],
        device='cuda'
    )

    space = {
        "target_weights_1": hp.uniform(label="target_weights_1", low=0.5, high=1.5),
        # "activation": hp.choice(label="activation", options=('relu', 'elu')),
        # "conv_layers": hp.choice(label="conv_layers", options=([16,]*4, [32,]*4)),
        "dropout": hp.uniform(label="dropout", low=0, high=0.5),
    }

    launch(args, space, max_evals=2, n_startup_jobs=1)
