"""
python test_model.py -r sim -m simple_cnn -t edge_2d
"""
import os
import itertools as it
import numpy as np
import pandas as pd

from tactile_data.tactile_servo_control import BASE_DATA_PATH, BASE_MODEL_PATH, BASE_RUNS_PATH
from tactile_image_processing.collect_data.setup_targets import POSE_LABEL_NAMES, SHEAR_LABEL_NAMES
from tactile_image_processing.collect_data.setup_targets import setup_targets
from tactile_image_processing.utils import load_json_obj, make_dir
from tactile_learning.supervised.models import create_model
from tactile_learning.utils.utils_plots import RegressionPlotter

from tactile_servo_control.utils.label_encoder import LabelEncoder
from tactile_servo_control.utils.labelled_model import LabelledModel
from tactile_servo_control.utils.parse_args import parse_args
from tactile_servo_control.utils.setup_embodiment import setup_embodiment


def test_model(
    robot,
    sensor,
    pose_model,
    collect_params,
    targets_df,
    preds_df,
    save_dir,
    image_dir,
    error_plotter
):
    # start 50mm above workframe origin
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.move_joints((*robot.joint_angles[:-1], 0))

    # drop 10mm to contact object
    clearance = (0, 0, 10, 0, 0, 0)
    robot.move_linear(np.zeros(6) - clearance)
    joint_angles = robot.joint_angles

    # ==== data testing loop ====
    for i, row in targets_df.iterrows():
        image_name = row.loc["sensor_image"]
        pose = row.loc[POSE_LABEL_NAMES].values.astype(float)
        shear = row.loc[SHEAR_LABEL_NAMES].values.astype(float)

        # report
        with np.printoptions(precision=2, suppress=True):
            print(f"\n\nCollecting for pose {i+1}: pose{pose}, shear{shear}")

        # move to above new pose (avoid changing pose in contact with object)
        robot.move_linear(pose + shear - clearance)

        # move down to offset pose
        robot.move_linear(pose + shear)

        # move to target pose inducing shear
        robot.move_linear(pose)

        # collect and process tactile image
        image_outfile = os.path.join(image_dir, image_name)
        tactile_image = sensor.process(image_outfile)
        preds_df.loc[i] = pose_model.predict(tactile_image)

        # move above the target pose
        robot.move_linear(pose - clearance)

        # if sorted, don't move to reset position
        if not collect_params['sort']:
            robot.move_joints(joint_angles)

    # save results
    preds_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    targets_df.to_csv(os.path.join(save_dir, 'targets.csv'), index=False)
    error_plotter.plot_interp = False
    error_plotter.final_plot(preds_df, targets_df)

    # finish 50mm above workframe origin then zero last joint
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.move_joints((*robot.joint_angles[:-1], 0))
    robot.close()


def testing(args):

    output_dir = '_'.join([args.robot, args.sensor])

    # test the trained networks
    for args.task, args.model, args.sample_num in it.product(args.tasks, args.models, args.sample_nums):

        model_dir_name = '_'.join(filter(None, [args.model, *args.model_version]))
        runs_dir_name = '_'.join(filter(None, [args.model, *args.run_version]))

        # setup save dir
        save_dir = os.path.join(BASE_RUNS_PATH, output_dir, args.task, runs_dir_name)
        image_dir = os.path.join(save_dir, "processed_images")
        make_dir(save_dir)
        make_dir(image_dir)

        # set data and model dir
        data_dir = os.path.join(BASE_DATA_PATH, output_dir, args.task, args.train_dirs[0])
        model_dir = os.path.join(BASE_MODEL_PATH, output_dir, args.task, model_dir_name)

        # load params
        collect_params = load_json_obj(os.path.join(data_dir, 'collect_params'))
        env_params = load_json_obj(os.path.join(model_dir, 'env_params'))
        model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
        model_image_params = load_json_obj(os.path.join(model_dir, 'model_image_params'))
        label_params = load_json_obj(os.path.join(model_dir, 'model_label_params'))
        if os.path.isfile(os.path.join(model_dir, 'processed_image_params.json')):
            sensor_image_params = load_json_obj(os.path.join(model_dir, 'processed_image_params'))
        else:
            sensor_image_params = load_json_obj(os.path.join(model_dir, 'sensor_image_params'))

        # create target_df
        targets_df = setup_targets(collect_params, args.sample_num, save_dir)
        preds_df = pd.DataFrame(columns=label_params['label_names'])

        # create the label encoder/decoder
        label_encoder = LabelEncoder(label_params, args.device)
        error_plotter = RegressionPlotter(label_params, save_dir, name='test_plot.png')

        # setup embodiment, network and model
        robot, sensor = setup_embodiment(
            env_params,
            sensor_image_params
        )

        # create the model
        model = create_model(
            in_dim=model_image_params['image_processing']['dims'],
            in_channels=1,
            out_dim=label_encoder.out_dim,
            model_params=model_params,
            saved_model_dir=model_dir,
            device=args.device
        )
        model.eval()

        pose_model = LabelledModel(
            model,
            model_image_params['image_processing'],
            label_encoder,
            args.device
        )

        test_model(
            robot,
            sensor,
            pose_model,
            collect_params,
            targets_df,
            preds_df,
            save_dir,
            image_dir,
            error_plotter
        )


if __name__ == "__main__":

    args = parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['edge_2d'],
        train_dirs=['train_data'],
        models=['simple_cnn'],
        model_version=[''],
        sample_nums=[100],
        run_version=[''],
        device='cuda'
    )

    testing(args)
