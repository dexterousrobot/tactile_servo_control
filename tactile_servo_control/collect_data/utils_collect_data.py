import os
import numpy as np
import pandas as pd

from tactile_gym.utils.general_utils import check_dir


def make_target_df_rand(
    num_poses,
    shuffle_data=False,
    pose_llims=[0, 0, 0, 0, 0, 0],
    pose_ulims=[0, 0, 0, 0, 0, 0],
    move_llims=[0, 0, 0, 0, 0, 0],
    move_ulims=[0, 0, 0, 0, 0, 0],
    obj_poses=[[0, 0, 0, 0, 0, 0]]
):

    # generate random poses
    np.random.seed()
    poses = np.random.uniform(
        low=pose_llims, high=pose_ulims, size=(num_poses, 6)
    )
    poses = poses[np.lexsort((poses[:, 1], poses[:, 5]))]
    moves = np.random.uniform(
        low=move_llims, high=move_ulims, size=(num_poses, 6)
    )

    # generate and save target data
    target_df = pd.DataFrame(
        columns=[
            "sensor_image",
            "obj_id",
            "obj_pose",
            "pose_id",
            "pose_1", "pose_2", "pose_3", "pose_4", "pose_5", "pose_6",
            "move_1", "move_2", "move_3", "move_4", "move_5", "move_6",
        ]
    )

    # populate dateframe
    for i in range(num_poses * len(obj_poses)):
        image_file = "image_{:d}.png".format(i + 1)
        i_pose, i_obj = (int(i % num_poses), int(i / num_poses))
        pose = poses[i_pose, :]
        move = moves[i_pose, :]
        target_df.loc[i] = np.hstack(
            ((image_file, i_obj + 1, obj_poses[i_obj], i_pose + 1), pose, move)
        )

    if shuffle_data:
        target_df = target_df.sample(frac=1).reset_index(drop=True)

    return target_df


def create_data_dir(
    collect_dir,
    target_df
):

    # set image dir and target file
    image_dir = os.path.join(collect_dir, "images")
    target_file = os.path.join(collect_dir, "targets.csv")

    # check save dir exists
    check_dir(collect_dir)
    os.makedirs(collect_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # save target csv
    target_df.to_csv(target_file, index=False)

    return image_dir
