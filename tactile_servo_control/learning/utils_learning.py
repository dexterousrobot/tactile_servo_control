# -*- coding: utf-8 -*-
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch

from tactile_learning.utils.utils_plots import LearningPlotter
from tactile_servo_control import BASE_MODEL_PATH

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sns.set_theme(style="darkgrid")

# tolerances for accuracy metrics
POS_TOL = 0.25  # mm
ROT_TOL = 1.0  # deg

# label names
POSE_LABEL_NAMES = ["x", "y", "z", "Rx", "Ry", "Rz"]
POS_LABEL_NAMES = ["x", "y", "z"]
ROT_LABEL_NAMES = ["Rx", "Ry", "Rz"]


def csv_row_to_label(row):
    return {
            'x': np.array(row['pose_1']),
            'y': np.array(row['pose_2']),
            'z': np.array(row['pose_3']),
            'Rx': np.array(row['pose_4']),
            'Ry': np.array(row['pose_5']),
            'Rz': np.array(row['pose_6']),
        }


class PoseEncoder:

    def __init__(self, 
                 label_names, 
                 pose_llims, 
                 pose_ulims, 
                 device='cuda'
        ):
        self.device = device
        self.label_names = label_names

        # create tensors for pose limits
        self.pose_llims_np = np.array(pose_llims)
        self.pose_ulims_np = np.array(pose_ulims)
        self.pose_llims_torch = torch.from_numpy(np.array(pose_llims)).float().to(self.device)
        self.pose_ulims_torch = torch.from_numpy(np.array(pose_ulims)).float().to(self.device)

    @property
    def out_dim(self):
        label_dims = [self.label_names.count(p) for p in POS_LABEL_NAMES] \
                    + [2*self.label_names.count(p) for p in ROT_LABEL_NAMES]
        return sum(label_dims)

    def encode_label(self, labels_dict):
        """
        Process raw pose data to NN friendly label for prediction.

        From -> {x, y, z, Rx, Ry, Rz}
        To   -> [norm(x), norm(y), norm(z), cos(Rx), sin(Rx), cos(Ry), sin(Ry), cos(Rz), sin(Rz)]
        """

        # encode pose to predictable label
        encoded_pose = []
        for label_name in self.label_names:

            # get the target from the dict
            target = labels_dict[label_name].float().to(self.device)

            # normalize pose label within limits
            if label_name in POS_LABEL_NAMES:
                llim = self.pose_llims_torch[POSE_LABEL_NAMES.index(label_name)]
                ulim = self.pose_ulims_torch[POSE_LABEL_NAMES.index(label_name)]
                norm_target = (((target - llim) / (ulim - llim)) * 2) - 1
                encoded_pose.append(norm_target.unsqueeze(dim=1))

            # sine/cosine encoding of angle
            if label_name in ROT_LABEL_NAMES:
                ang = target * np.pi/180
                encoded_pose.append(torch.sin(ang).float().to(self.device).unsqueeze(dim=1))
                encoded_pose.append(torch.cos(ang).float().to(self.device).unsqueeze(dim=1))

        # combine targets to make one label tensor
        labels = torch.cat(encoded_pose, 1)

        return labels

    def decode_label(self, outputs):
        """
        Process NN predictions to raw pose data, always decodes to cpu.

        From  -> [norm(x), norm(y), norm(z), cos(Rx), sin(Rx), cos(Ry), sin(Ry), cos(Rz), sin(Rz)]
        To    -> {x, y, z, Rx, Ry, Rz}
        """

        # decode preictable label to pose
        decoded_pose = {
            'x': torch.zeros(outputs.shape[0]),
            'y': torch.zeros(outputs.shape[0]),
            'z': torch.zeros(outputs.shape[0]),
            'Rx': torch.zeros(outputs.shape[0]),
            'Ry': torch.zeros(outputs.shape[0]),
            'Rz': torch.zeros(outputs.shape[0]),
        }

        label_name_idx = 0
        for label_name in self.label_names:

            if label_name in POS_LABEL_NAMES:
                predictions = outputs[:, label_name_idx].detach().cpu()
                llim = self.pose_llims_np[POSE_LABEL_NAMES.index(label_name)]
                ulim = self.pose_ulims_np[POSE_LABEL_NAMES.index(label_name)]
                decoded_predictions = (((predictions + 1) / 2) * (ulim - llim)) + llim
                decoded_pose[label_name] = decoded_predictions
                label_name_idx += 1

            if label_name in ROT_LABEL_NAMES:
                sin_predictions = outputs[:, label_name_idx].detach().cpu()
                cos_predictions = outputs[:, label_name_idx + 1].detach().cpu()
                pred_rot = torch.atan2(sin_predictions, cos_predictions)
                pred_rot = pred_rot * (180.0 / np.pi)
                decoded_pose[label_name] = pred_rot
                label_name_idx += 2

        return decoded_pose

    def calc_batch_metrics(self, labels, predictions):
        """
        Calculate metrics useful for measuring progress throughout training.

        Returns: dict of metrics
            {
                'metric': np.array()
            }
        """
        err_df = self.err_metric(labels, predictions)
        acc_df = self.acc_metric(err_df)
        return err_df, acc_df

    def err_metric(self, labels, predictions):
        """
        Error metric for regression problem, returns dict of errors in interpretable units.
        Position error (mm), Rotation error (degrees).
        """
        err_df = pd.DataFrame(columns=POSE_LABEL_NAMES)
        for label_name in self.label_names:

            if label_name in POS_LABEL_NAMES:
                abs_err = torch.abs(
                    labels[label_name] - predictions[label_name]
                ).detach().cpu().numpy()

            if label_name in ROT_LABEL_NAMES:
                # convert rad
                targ_rot = labels[label_name] * np.pi/180
                pred_rot = predictions[label_name] * np.pi/180

                # Calculate angle difference, taking into account periodicity (thanks ChatGPT)
                abs_err = torch.abs(
                    torch.atan2(torch.sin(targ_rot - pred_rot), torch.cos(targ_rot - pred_rot))
                ).detach().cpu().numpy() * (180.0 / np.pi)

            err_df[label_name] = abs_err

        return err_df

    def acc_metric(self, err_df):
        """
        Accuracy metric for regression problem, counting the number of predictions within a tolerance.
        Position Tolerance (mm), Rotation Tolerance (degrees)
        """

        batch_size = err_df.shape[0]
        acc_df = pd.DataFrame(columns=[*POSE_LABEL_NAMES, 'overall_acc'])
        overall_correct = np.ones(batch_size, dtype=bool)
        for label_name in self.label_names:

            if label_name in POS_LABEL_NAMES:
                abs_err = err_df[label_name]
                correct = (abs_err < POS_TOL)

            if label_name in ROT_LABEL_NAMES:
                abs_err = err_df[label_name]
                correct = (abs_err < ROT_TOL)

            overall_correct = overall_correct & correct
            acc_df[label_name] = correct.astype(np.float32)

        # count where all predictions are correct for overall accuracy
        acc_df['overall_acc'] = overall_correct.astype(np.float32)

        return acc_df


class ErrorPlotter:
    def __init__(
        self,
        label_names,
        save_dir=None,
        plot_during_training=False,
        name="error_plot.png"
    ):
        self._label_names = label_names
        self._save_dir = save_dir
        self._name = name
        self.plot_during_training = plot_during_training

        if plot_during_training:
            plt.ion()
            self._fig, self._axs = plt.subplots(2, 3, figsize=(12, 7))
            self._fig.subplots_adjust(wspace=0.3)

    def update(
        self,
        pred_df,
        targ_df,
        err_df,
    ):

        for ax in self._axs.flat:
            ax.clear()

        n_smooth = int(pred_df.shape[0] / 20)

        for i, ax in enumerate(self._axs.flat):

            pose_label = POSE_LABEL_NAMES[i]

            # skip labels we are not actively trying to predict
            if pose_label not in self._label_names:
                continue

            # sort all dfs by target
            targ_df = targ_df.sort_values(by=pose_label)

            pred_df = pred_df.assign(temp=targ_df[pose_label])
            pred_df = pred_df.sort_values(by='temp')
            pred_df = pred_df.drop('temp', axis=1)

            err_df = err_df.assign(temp=targ_df[pose_label])
            err_df = err_df.sort_values(by='temp')
            err_df = err_df.drop('temp', axis=1)

            ax.scatter(
                targ_df[pose_label], pred_df[pose_label], s=1,
                c=err_df[pose_label], cmap="inferno"
            )
            ax.plot(
                targ_df[pose_label].rolling(n_smooth).mean(),
                pred_df[pose_label].rolling(n_smooth).mean(),
                linewidth=2, c='r'
            )
            ax.set(xlabel=f"target {pose_label}", ylabel=f"predicted {pose_label}")

            pose_llim = np.round(min(targ_df[pose_label]))
            pose_ulim = np.round(max(targ_df[pose_label]))
            ax.set_xlim(pose_llim, pose_ulim)
            ax.set_ylim(pose_llim, pose_ulim)

            ax.text(0.05, 0.9, 'MAE = {:.4f}'.format(err_df[pose_label].mean()), transform=ax.transAxes)
            ax.grid(True)

        if self._save_dir is not None:
            save_file = os.path.join(self._save_dir, self._name)
            self._fig.savefig(save_file, dpi=320, pad_inches=0.01, bbox_inches='tight')

        self._fig.canvas.draw()
        plt.pause(0.01)

    def final_plot(
        self,
        pred_df,
        targ_df,
        err_df,
    ):
        if not self.plot_during_training:
            self._fig, self._axs = plt.subplots(2, 3, figsize=(12, 7))
            self._fig.subplots_adjust(wspace=0.3)

        self.update(
            pred_df,
            targ_df,
            err_df
        )
        plt.show()


if __name__ == '__main__':

    from setup_learning import setup_task

    # task = 'surface_3d'
    task = 'edge_2d'
    # task = 'edge_3d'
    # task = 'edge_5d'

    out_dim, label_names = setup_task(task)

    model = 'simple_cnn'

    # path to model for loading
    save_dir = os.path.join(BASE_MODEL_PATH, 'sim', task, model)

    # load and plot predictions
    with open(os.path.join(save_dir, 'val_pred_targ_err.pkl'), 'rb') as f:
        pred_df, targ_df, err_df, label_names = pickle.load(f)

    error_plotter = ErrorPlotter(save_dir=save_dir, label_names=label_names, name='val_error_plot.png')
    error_plotter.final_plot(pred_df, targ_df, err_df)

    # load and plot training
    with open(os.path.join(save_dir, 'train_val_loss_acc.pkl'), 'rb') as f:
        train_loss, val_loss, train_acc, val_acc = pickle.load(f)

    learning_plotter = LearningPlotter(save_dir=save_dir)
    learning_plotter.final_plot(train_loss, val_loss, train_acc, val_acc)
