"""
python test_model.py -r sim -m simple_cnn -t edge_2d
"""
import os
import itertools as it
import numpy as np
import pandas as pd

from tactile_data.tactile_servo_control import BASE_MODEL_PATH, BASE_RUNS_PATH
from tactile_data.utils_data import load_json_obj, make_dir
from tactile_learning.supervised.models import create_model

from tactile_servo_control.collect_data.utils_collect_data import setup_target_df
from tactile_servo_control.learning.utils_learning import LabelEncoder, LabelledModel
from tactile_servo_control.learning.utils_plots import RegressErrorPlotter
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
        pose = row.loc[collect_params['pose_label_names']].values.astype(float)
        shear = row.loc[collect_params['shear_label_names']].values.astype(float)

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
    error_plotter.final_plot(preds_df, targets_df)

    # finish 50mm above workframe origin then zero last joint 
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.move_joints((*robot.joint_angles[:-1], 0))
    robot.close()


if __name__ == "__main__":

    args = parse_args(
        robot='sim', 
        sensor='tactip',
        tasks=['edge_2d'],
        models=['simple_cnn'],
        version=['test'],
        device='cuda'
    )

    num_poses = 100

    # test the trained networks
    for args.task, args.model in it.product(args.tasks, args.models):

        output_dir = '_'.join([args.robot, args.sensor])
        model_dir_name = '_'.join([args.model, *args.version]) 

        #  setup save dir
        save_dir = os.path.join(BASE_RUNS_PATH, output_dir, args.task, model_dir_name)
        image_dir = os.path.join(save_dir, "processed_images")
        make_dir(save_dir)
        make_dir(image_dir)

        # set data and model dir
        model_dir = os.path.join(BASE_MODEL_PATH, output_dir, args.task, model_dir_name)

        # load params
        collect_params = load_json_obj(os.path.join(model_dir, 'collect_params'))
        env_params = load_json_obj(os.path.join(model_dir, 'env_params'))
        model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
        preproc_params = load_json_obj(os.path.join(model_dir, 'preproc_params'))
        sensor_params = load_json_obj(os.path.join(model_dir, 'sensor_params'))
        task_params = load_json_obj(os.path.join(model_dir, 'task_params'))

        # create target_df
        targets_df = setup_target_df(collect_params, num_poses, save_dir) 
        preds_df = pd.DataFrame(columns=task_params['label_names'])

        # create the label encoder/decoder
        label_encoder = LabelEncoder(task_params, args.device)
        error_plotter = RegressErrorPlotter(task_params, save_dir, name='test_plot.png', plot_interp=False)

        # setup embodiment, network and model
        robot, sensor = setup_embodiment(
            env_params, 
            sensor_params
        )

        # create the model
        model = create_model(
            in_dim=preproc_params['image_processing']['dims'],
            in_channels=1,
            out_dim=label_encoder.out_dim,
            model_params=model_params,
            saved_model_dir=model_dir,
            device=args.device
        )
        model.eval()

        pose_model = LabelledModel(
            model,
            preproc_params['image_processing'],
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
            error_plotter
        )
