import os
import numpy as np

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def collect_data(
    robot, 
    sensor,
    targets_df,
    image_dir,
    task_params,
):

    # start 50mm above workframe origin with zero joint 6
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.move_joints([*robot.joint_angles[:-1], 0])

    # collect reference image
    image_outfile = os.path.join(image_dir, 'image_0.png')
    sensor.process(image_outfile)

    # clear object by 10mm; use as reset pose 
    clearance = (0, 0, 10, 0, 0, 0)
    robot.move_linear(np.zeros(6) - clearance)
    joint_angles = robot.joint_angles

    # ==== data collection loop ====
    for i, row in targets_df.iterrows():
        image_name = row.loc["sensor_image"]
        pose = row.loc[task_params['pose_label_names']].values.astype(float)
        shear = row.loc[task_params['shear_label_names']].values.astype(float)
        # if i<228: continue

        # report
        with np.printoptions(precision=1, suppress=True):
            print(f"Collecting for pose {i+1}: pose{pose}, shear{shear}")

        # move to above new pose (avoid changing pose in contact with object)
        robot.move_linear(pose + shear - clearance)
 
        # move down to offset pose
        robot.move_linear(pose + shear)

        # move to target pose inducing shear
        robot.move_linear(pose)

        # collect and process tactile image
        image_outfile = os.path.join(image_dir, image_name)
        sensor.process(image_outfile)

        # move above the target pose
        robot.move_linear(pose - clearance)

        # if sorted, don't move to reset position
        if not task_params['sort']:
            robot.move_joints(joint_angles)

    # finish 50mm above workframe origin then zero last joint 
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.move_joints((*robot.joint_angles[:-1], 0))
    robot.close()


if __name__ == "__main__":
    pass
