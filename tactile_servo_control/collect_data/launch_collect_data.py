"""
python launch_collect_data.py -r sim -s tactip -t edge_5d
"""
import os
import numpy as np

from tactile_data.tactile_servo_control import BASE_DATA_PATH
from tactile_data.utils_data import make_dir

from tactile_servo_control.collect_data.setup_collect_data import setup_collect_data
from tactile_servo_control.collect_data.utils_collect_data import setup_target_df
from tactile_servo_control.utils.setup_embodiment import setup_embodiment
from tactile_servo_control.utils.parse_args import parse_args


def collect_data(
    robot,
    sensor,
    targets_df,
    image_dir,
    collect_params,
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
        pose = row.loc[collect_params['pose_label_names']].values.astype(float)
        shear = row.loc[collect_params['shear_label_names']].values.astype(float)

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
        if not collect_params['sort']:
            robot.move_joints(joint_angles)

    # finish 50mm above workframe origin then zero last joint
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.move_joints((*robot.joint_angles[:-1], 0))
    robot.close()


def launch():

    args = parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['edge_2d'],
        version=['']
    )

    data_params = {
        'data': 500,
    }

    for args.task in args.tasks:
        for data_dir_name, num_samples in data_params.items():

            data_dir_name = '_'.join(filter(None, [data_dir_name, *args.version]))
            output_dir = '_'.join([args.robot, args.sensor])

            # setup save dir
            save_dir = os.path.join(BASE_DATA_PATH, output_dir, args.task, data_dir_name)
            image_dir = os.path.join(save_dir, "images")
            make_dir(save_dir)
            make_dir(image_dir)

            # setup parameters
            collect_params, env_params, sensor_params = setup_collect_data(
                args.robot,
                args.sensor,
                args.task,
                save_dir
            )

            # setup embodiment
            robot, sensor = setup_embodiment(
                env_params,
                sensor_params
            )

            # setup targets to collect
            target_df = setup_target_df(
                collect_params,
                num_samples,
                save_dir
            )

            # collect
            collect_data(
                robot,
                sensor,
                target_df,
                image_dir,
                collect_params
            )


if __name__ == "__main__":
    launch()
