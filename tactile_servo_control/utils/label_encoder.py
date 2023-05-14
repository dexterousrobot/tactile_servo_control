"""
python utils_learning.py -r cr -s tactip_331 -m simple_cnn -t edge_2d
"""
import os
import pickle
import itertools as it
import numpy as np
import pandas as pd
import torch

from tactile_data.tactile_servo_control import BASE_MODEL_PATH
from tactile_image_processing.utils import load_json_obj
from tactile_learning.utils.utils_plots import LearningPlotter, RegressionPlotter

from tactile_servo_control.utils.parse_args import parse_args


class LabelEncoder:

    def __init__(self, task_params, device='cuda'):
        self.device = device
        self.label_names = task_params['label_names']
        self.target_label_names = list(filter(None, task_params['target_label_names']))
        num_targets = len(self.target_label_names)

        # optional arguments
        self.periodic_label_names = task_params.get('periodic_label_names', [])
        self.target_weights = task_params.get('target_weights', np.ones(num_targets))
        self.tolerences = task_params.get('tolerences', np.ones(num_targets))

        # create tensors for pose limits
        self.llims_np = np.array(task_params['llims'])
        self.ulims_np = np.array(task_params['ulims'])
        self.llims_torch = torch.from_numpy(self.llims_np).float().to(self.device)
        self.ulims_torch = torch.from_numpy(self.ulims_np).float().to(self.device)

    @property
    def out_dim(self):
        periodic_dims = [self.target_label_names.count(p) for p in self.periodic_label_names]
        return len(list(filter(None, self.target_label_names))) + sum(periodic_dims)

    def encode_norm(self, target, label_name):
        llim = self.llims_torch[self.label_names.index(label_name)]
        ulim = self.ulims_torch[self.label_names.index(label_name)]
        norm_target = (((target - llim) / (ulim - llim)) * 2) - 1
        return norm_target.unsqueeze(dim=1)

    def decode_norm(self, prediction, label_name):
        llim = self.llims_np[self.label_names.index(label_name)]
        ulim = self.ulims_np[self.label_names.index(label_name)]
        return (((prediction + 1) / 2) * (ulim - llim)) + llim

    def encode_circnorm(self, target):
        ang = target * np.pi/180
        return [torch.sin(ang).float().to(self.device).unsqueeze(dim=1),
                torch.cos(ang).float().to(self.device).unsqueeze(dim=1)]

    def decode_circnorm(self, vec_prediction):
        pred_rot = torch.atan2(*vec_prediction)
        return pred_rot * 180/np.pi

    def encode_label(self, labels_dict):
        """
        Process raw pose data to NN friendly label for prediction.
        Default: maps to weight * range(-1,+1)
        Periodic: maps to weight * [cos angle, sin angle]
        """

        # encode pose to predictable label
        encoded_pose = []
        # target_label_names = filter(None, self.target_label_names)
        for label_name, weight in zip(self.target_label_names, self.target_weights):

            # get the target from the dict
            target = labels_dict[label_name].float().to(self.device)

            # normalize pose label within limits
            if label_name not in self.periodic_label_names:
                encoded_pose.append(weight * self.encode_norm(target, label_name))

            # if periodic use sine/cosine encoding of angle
            if label_name in self.periodic_label_names:
                encoded_pose.extend([weight * p for p in self.encode_circnorm(target)])

        return torch.cat(encoded_pose, 1)

    def decode_label(self, outputs):
        """
        Process NN predictions to raw pose data, always decodes to cpu.
        Inverse of encode
        """

        # decode predicted label to pose
        decoded_pose = {label: torch.zeros(outputs.shape[0]) for label in self.label_names}
        # target_label_names = filter(None, self.target_label_names)

        ind = 0
        for label_name, weight in zip(self.target_label_names, self.target_weights):

            if label_name not in self.periodic_label_names:
                prediction = outputs[:, ind].detach().cpu() / weight
                decoded_pose[label_name] = self.decode_norm(prediction, label_name)
                ind += 1

            elif label_name in self.periodic_label_names:
                vec_prediction = [outputs[:, ind].detach().cpu() / weight,
                                  outputs[:, ind+1].detach().cpu() / weight]
                decoded_pose[label_name] = self.decode_circnorm(vec_prediction)
                ind += 2

        return decoded_pose

    def print_metrics(self, metrics):
        """
        Formatted print of metrics given by calc_metrics.
        """
        err_df, acc_df = metrics['err'], metrics['acc']
        print('Error: ')
        print(err_df[self.target_label_names].mean())
        print('Accuracy: ')
        print(acc_df[self.target_label_names].mean())

    def write_metrics(self, writer, metrics, epoch, mode='val'):
        """
        Write metrics given by calc_metrics to tensorboard.
        """
        err_df, acc_df = metrics['err'], metrics['acc']
        for label_name in self.target_label_names:
            writer.add_scalar(f'accuracy/{mode}/{label_name}', acc_df[label_name].mean(), epoch)
            writer.add_scalar(f'loss/{mode}/{label_name}', err_df[label_name].mean(), epoch)

    def calc_metrics(self, labels, predictions):
        """
        Calculate metrics useful for measuring progress throughout training.
        """
        err_df = self.err_metric(labels, predictions)
        acc_df = self.acc_metric(err_df)
        metrics = {
            'err': err_df,
            'acc': acc_df
        }
        return metrics

    def err_metric(self, labels, predictions):
        """
        Error metric for regression problem, returns df of errors.
        """
        err_df = pd.DataFrame(columns=self.label_names)
        for label_name in filter(None, self.target_label_names):
            if label_name not in self.periodic_label_names:
                abs_err = np.abs(
                    labels[label_name] - predictions[label_name]
                )

            elif label_name in self.periodic_label_names:
                targ_rot = labels[label_name] * np.pi/180
                pred_rot = predictions[label_name] * np.pi/180

                # Calculate angle difference, taking into account periodicity (thanks ChatGPT)
                abs_err = np.abs(
                    np.arctan2(np.sin(targ_rot - pred_rot), np.cos(targ_rot - pred_rot))
                ) * 180/np.pi

            err_df[label_name] = abs_err

        return err_df

    def acc_metric(self, err_df):
        """
        Accuracy metric for regression problem, counting the number of predictions within a tolerance.
        Returns df of accuracies.
        """

        batch_size = err_df.shape[0]
        acc_df = pd.DataFrame(columns=[*self.label_names, 'overall_acc'])
        overall_correct = np.ones(batch_size, dtype=bool)

        for label_name, tolerence in zip(filter(None, self.target_label_names), self.tolerences):
            abs_err = err_df[label_name]
            correct = (abs_err < tolerence)

            overall_correct = overall_correct & correct
            acc_df[label_name] = correct.astype(np.float32)

        # count where all predictions are correct for overall accuracy
        acc_df['overall_acc'] = overall_correct.astype(np.float32)

        return acc_df


if __name__ == '__main__':

    args = parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['edge_2d'],
        models=['simple_cnn'],
        version=[''],
        device='cuda'
    )

    for args.task, args.model in it.product(args.tasks, args.models):

        output_dir = '_'.join([args.robot, args.sensor])
        model_dir_name = '_'.join(filter(None, [args.model, *args.version]))

        # set save dir
        save_dir = os.path.join(BASE_MODEL_PATH, output_dir, args.task, model_dir_name)

        # create task params
        task_params = load_json_obj(os.path.join(save_dir, 'task_params'))

        # load and plot predictions
        with open(os.path.join(save_dir, 'val_pred_targ_err.pkl'), 'rb') as f:
            pred_df, targ_df, err_df, label_names = pickle.load(f)

        error_plotter = RegressionPlotter(task_params, save_dir, 'error_plot_best.png')
        error_plotter.final_plot(pred_df, targ_df, err_df)

        # load and plot training
        with open(os.path.join(save_dir, 'train_val_loss_acc.pkl'), 'rb') as f:
            train_loss, val_loss, train_acc, val_acc = pickle.load(f)

        learning_plotter = LearningPlotter(save_dir=save_dir)
        learning_plotter.final_plot(train_loss, val_loss, train_acc, val_acc)
