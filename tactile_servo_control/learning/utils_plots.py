import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


class ErrorPlotter:
    def __init__(
        self,
        task_params,
        save_dir=None,
        name="error_plot.png",
        plot_during_training=False,
    ):
        self.pose_label_names = task_params['pose_label_names']
        self.target_label_names = task_params['target_label_names']
        self.pose_llims = task_params['pose_llims']
        self.pose_ulims = task_params['pose_ulims']
        self.save_dir = save_dir
        self.name = name
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

            label_name = self.pose_label_names[i]

            # skip labels we are not actively trying to predict
            if label_name not in self.target_label_names:
                continue

            # sort all dfs by target
            targ_df = targ_df.sort_values(by=label_name)

            pred_df = pred_df.assign(temp=targ_df[label_name])
            pred_df = pred_df.sort_values(by='temp')
            pred_df = pred_df.drop('temp', axis=1)

            err_df = err_df.assign(temp=targ_df[label_name])
            err_df = err_df.sort_values(by='temp')
            err_df = err_df.drop('temp', axis=1)

            ax.scatter(
                targ_df[label_name], pred_df[label_name], s=1,
                c=err_df[label_name], cmap="inferno"
            )
            ax.plot(
                targ_df[label_name].rolling(n_smooth).mean(),
                pred_df[label_name].rolling(n_smooth).mean(),
                linewidth=2, c='r'
            )
            ax.set(xlabel=f"target {label_name}", ylabel=f"predicted {label_name}")

            ax.set_xlim(self.pose_llims[i], self.pose_ulims[i])
            ax.set_ylim(self.pose_llims[i], self.pose_ulims[i])

            ax.text(0.05, 0.9, 'MAE = {:.4f}'.format(err_df[label_name].mean()), transform=ax.transAxes)
            ax.grid(True)

        if self.save_dir is not None:
            save_file = os.path.join(self.save_dir, self.name)
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
            pred_df, targ_df, err_df
        )
        plt.show()


if __name__ == '__main__':
    pass
