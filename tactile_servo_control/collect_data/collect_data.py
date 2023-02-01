"""
python collect_data.py -t surface_3d
python collect_data.py -t edge_2d
python collect_data.py -t edge_3d
python collect_data.py -t edge_5d
python collect_data.py -t surface_3d edge_2d edge_3d edge_5d
"""

import os
import argparse
import numpy as np

from tactile_servo_control import BASE_DATA_PATH

from tactile_servo_control.utils_robot_sim.setup_embodiment_env import setup_embodiment_env
from tactile_servo_control.collect_data.setup_collect_sim_data import setup_collect_data

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
np.set_printoptions(precision=1, suppress=True)


def collect_data(
    embodiment,
    target_df,
    image_dir
):
    # start above workframe origin
    hover = embodiment.hover
    embodiment.move_linear(hover)

    # ==== data collection loop ====
    for _, row in target_df.iterrows():
        i_obj = int(row.loc["obj_id"])
        i_pose = int(row.loc["pose_id"])
        pose = row.loc["pose_1":"pose_6"].values.astype(np.float32)
        move = row.loc["move_1":"move_6"].values.astype(np.float32)
        obj_pose = row.loc["obj_pose"]
        sensor_image = row.loc["sensor_image"]

        # report
        print(f"Collecting data for object {i_obj}, pose {i_pose}: pose{pose}, move{move}")

        # pose is relative to object
        pose += obj_pose

        # move to above new pose (avoid changing pose in contact with object)
        embodiment.move_linear(np.array(pose) - np.array(move) + np.array(hover))

        # move down to offset position
        embodiment.move_linear(np.array(pose) - np.array(move))

        # move to target positon inducing shear effects
        embodiment.move_linear(np.array(pose))

        # process tactile image
        image_outfile = os.path.join(image_dir, sensor_image)
        embodiment.sensor_process(outfile=image_outfile)

        # move to target positon inducing shear effects
        embodiment.move_linear(np.array(pose) + np.array(hover))

    # finish above workframe origin
    embodiment.move_linear(hover)
    embodiment.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose task from [surface_3d edge_2d edge_3d edge_5d].",
        default=['edge_2d']
    )

    # parse arguments
    args = parser.parse_args()
    tasks = args.tasks

    for task in tasks:

        collect_dir = os.path.join(
                BASE_DATA_PATH, task, 'data'
            )

        target_df, image_dir, env_params, sensor_params = \
            setup_collect_data[task](
                collect_dir
            )

        embodiment = setup_embodiment_env(
            **env_params,
            sensor_params=sensor_params,
            show_gui=True,  # quick_mode=True
        )

        collect_data(
            embodiment,
            target_df,
            image_dir
        )
