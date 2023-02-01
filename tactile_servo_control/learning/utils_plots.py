import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tactile_servo_control import BASE_MODEL_PATH
from tactile_servo_control.learning.utils_learning import POSE_LABEL_NAMES
from tactile_servo_control.learning.setup_learning import setup_task

from tactile_learning.utils.utils_plots import LearningPlotter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sns.set_theme(style="darkgrid")


class ErrorPlotter:
    def __init__(
        self,
        target_label_names,
        save_dir=None,
        plot_during_training=False,
        name="error_plot.png"
    ):
        self._target_label_names = target_label_names
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
            if pose_label not in self._target_label_names:
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

    # task = 'surface_3d'
    task = 'edge_2d'
    # task = 'edge_3d'
    # task = 'edge_5d'

    out_dim, label_names = setup_task(task)

    model = 'simple_cnn'

    # path to model for loading
    save_dir = os.path.join(BASE_MODEL_PATH, task, model)

    # load and plot predictions
    with open(os.path.join(save_dir, 'val_pred_targ_err.pkl'), 'rb') as f:
        pred_df, targ_df, err_df, label_names = pickle.load(f)

    error_plotter = ErrorPlotter(save_dir=save_dir, target_label_names=label_names, name='val_error_plot.png')
    error_plotter.final_plot(pred_df, targ_df, err_df)

    # load and plot training
    with open(os.path.join(save_dir, 'train_val_loss_acc.pkl'), 'rb') as f:
        train_loss, val_loss, train_acc, val_acc = pickle.load(f)

    learning_plotter = LearningPlotter(save_dir=save_dir)
    learning_plotter.final_plot(train_loss, val_loss, train_acc, val_acc)
