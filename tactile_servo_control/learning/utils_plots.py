import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


class RegressErrorPlotter:
    def __init__(
        self,
        task_params,
        save_dir=None,
        name="error_plot.png",
        plot_during_training=False,
        plot_interp=True
    ):
        self.target_label_names = task_params['target_label_names']
        self.save_dir = save_dir
        self.name = name
        self.plot_during_training = plot_during_training
        self.plot_interp = plot_interp

        self.n_plots = len(self.target_label_names)
        self.n_rows = int(np.ceil(self.n_plots/3))
        self.n_cols = np.minimum(self.n_plots, 3)

        if self.n_plots == 2 or self.n_plots == 5:
            self.target_label_names.insert(2, None)

        if plot_during_training:
            plt.ion()
            self._fig, self._axs = plt.subplots(self.n_rows, self.n_cols,
                                                figsize=(4*self.n_cols, 3.5*self.n_rows))
            self._fig.subplots_adjust(wspace=0.3)

    def update(
        self,
        pred_df,
        targ_df,
        metrics=None,
    ):
        for ax in self._axs.flat:
            ax.clear()

        n_smooth = int(pred_df.shape[0] / 20)

        err_df = metrics['err']

        for ax, label_name in zip(self._axs.flat, self.target_label_names):
            if label_name:

                targ_df = targ_df.sort_values(by=label_name)

                pred_df = pred_df.assign(temp=targ_df[label_name])
                pred_df = pred_df.sort_values(by='temp')
                pred_df = pred_df.drop('temp', axis=1)

                if isinstance(err_df, pd.DataFrame):
                    err_df = err_df.assign(temp=targ_df[label_name])
                    err_df = err_df.sort_values(by='temp')
                    err_df = err_df.drop('temp', axis=1)

                    ax.scatter(
                        targ_df[label_name].astype(float),
                        pred_df[label_name].astype(float),
                        s=1, c=err_df[label_name], cmap="inferno"
                    )

                else:
                    ax.scatter(
                        targ_df[label_name].astype(float),
                        pred_df[label_name].astype(float), s=1, c='k'
                    )

                if self.plot_interp:
                    ax.plot(
                        targ_df[label_name].astype(float).rolling(n_smooth).mean(),
                        pred_df[label_name].astype(float).rolling(n_smooth).mean(),
                        linewidth=2, c='r'
                    )

                ax.set(xlabel=f"target {label_name}", ylabel=f"predicted {label_name}")
                xlim = (
                    np.round(targ_df[label_name].astype(float).min()),
                    np.round(targ_df[label_name].astype(float).max())
                )
                xticks = ax.get_xticks()
                ax.set_xticks(xticks), ax.set_yticks(xticks)
                ax.set_xlim(*xlim), ax.set_ylim(*xlim)

                if isinstance(err_df, pd.DataFrame):
                    ax.text(0.05, 0.9, 'MAE = {:.4f}'.format(err_df[label_name].mean()), transform=ax.transAxes)
                ax.grid(True)

            else:
                ax.axis('off')

        if self.save_dir is not None:
            save_file = os.path.join(self.save_dir, self.name)
            self._fig.savefig(save_file, dpi=320, pad_inches=0.01, bbox_inches='tight')

        self._fig.canvas.draw()
        plt.pause(0.01)

    def final_plot(
        self,
        pred_df,
        targ_df,
        metrics=None,
    ):
        if not self.plot_during_training:
            self._fig, self._axs = plt.subplots(self.n_rows, self.n_cols,
                                                figsize=(4*self.n_cols, 3.5*self.n_rows))
            self._fig.subplots_adjust(wspace=0.3)

        self.update(pred_df, targ_df, metrics)
        plt.show()


if __name__ == '__main__':
    pass
