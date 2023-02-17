import os
import numpy as np
import pandas as pd
import torch

from tactile_learning.utils.utils_learning import save_json_obj, load_json_obj

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


def get_pose_limits(data_dirs, save_dir):
    """
     Get limits for poses of data collected, used to encode/decode pose for prediction
     data_dirs is expected to be a list of data directories

     When using more than one data source, limits are taken at the extremes of those used for collection.
    """
    pose_llims, pose_ulims = [], []
    for data_dir in data_dirs:
        pose_params = load_json_obj(os.path.join(data_dir, 'pose_params'))
        pose_llims.append(pose_params['pose_llims'])
        pose_ulims.append(pose_params['pose_ulims'])

    pose_llims = np.min(pose_llims, axis=0)
    pose_ulims = np.max(pose_ulims, axis=0)

    # save limits
    pose_limits = {
        'pose_llims': list(pose_llims*1.0),
        'pose_ulims': list(pose_ulims*1.0),
    }

    save_json_obj(pose_limits, os.path.join(save_dir, 'pose_params'))

    return pose_llims, pose_ulims


class PoseEncoder:

    def __init__(self, target_label_names, pose_limits, device):
        self.device = device
        self.target_label_names = target_label_names

        # create tensors for pose limits
        self.pose_llims_np = np.array(pose_limits[0])
        self.pose_ulims_np = np.array(pose_limits[1])
        self.pose_llims_torch = torch.from_numpy(np.array(pose_limits[0])).float().to(self.device)
        self.pose_ulims_torch = torch.from_numpy(np.array(pose_limits[1])).float().to(self.device)

    def encode_label(self, labels_dict):
        """
        Process raw pose data to NN friendly label for prediction.

        From -> {x, y, z, Rx, Ry, Rz}
        To   -> [norm(x), norm(y), norm(z), cos(Rx), sin(Rx), cos(Ry), sin(Ry), cos(Rz), sin(Rz)]
        """

        # encode pose to predictable label
        encoded_pose = []
        for label_name in self.target_label_names:

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
        for label_name in self.target_label_names:

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
        for label_name in self.target_label_names:

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
        for label_name in self.target_label_names:

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
