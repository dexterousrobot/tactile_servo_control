# -*- coding: utf-8 -*-
import os
import numpy as np

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def collect_data(
    embodiment,
    target_df,
    image_dir
):
    # start 50mm above workframe origin
    embodiment.move_linear([0, 0, -50, 0, 0, 0])

    # collect reference image
    image_outfile = os.path.join(image_dir, 'image_0.png')
    embodiment.sensor.process(image_outfile)

    # drop 10mm to contact object
    tap_move = [0, 0, 10, 0, 0, 0]

    # ==== data collection loop ====
    for i, row in target_df.iterrows():
        i_obj = int(row.loc["obj_id"])
        i_pose = int(row.loc["pose_id"])
        pose = row.loc["pose_1":"pose_6"].values.astype(np.float32)
        move = row.loc["move_1":"move_6"].values.astype(np.float32)
        obj_pose = row.loc["obj_pose"]
        sensor_image = row.loc["sensor_image"]
        # if i < 2650: continue

        # report
        with np.printoptions(precision=1, suppress=True):
            print(f"Collecting for object {i_obj}, pose {i_pose}: pose{pose}, move{move}")

        # pose is relative to object
        pose += obj_pose

        # move to above new pose (avoid changing pose in contact with object)
        embodiment.move_linear(pose + move - tap_move)
 
        # move down to offset position
        embodiment.move_linear(pose + move)

        # move to target positon inducing shear effects
        embodiment.move_linear(pose)

        # collect and process tactile image
        image_outfile = os.path.join(image_dir, sensor_image)
        embodiment.sensor.process(image_outfile)

        # move to target positon inducing shear effects
        embodiment.move_linear(pose - tap_move)

    # finish 50mm above workframe origin then zero last joint 
    embodiment.move_linear([0, 0, -50, 0, 0, 0])
    embodiment.move_joints([*embodiment.joint_angles[:-1], 0])

    embodiment.close()


if __name__ == "__main__":
    pass
