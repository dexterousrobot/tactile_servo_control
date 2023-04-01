"""
python test_model.py -r sim -m simple_cnn -t edge_2d
"""
import os
import numpy as np
import pandas as pd

from tactile_data.tactile_servo_control import BASE_MODEL_PATH, BASE_RUNS_PATH
from tactile_data.utils_data import load_json_obj, make_dir
from tactile_learning.supervised.models import create_model
from tactile_servo_control.collect_data.utils_collect_data import setup_target_df
from tactile_servo_control.utils.setup_parse_args import setup_parse_args
from tactile_servo_control.utils.setup_embodiment import setup_embodiment

from utils_learning import LabelEncoder, LabelledModel
from utils_plots import RegressErrorPlotter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def test_model(
    robot,
    sensor,
    pose_model,
    collect_params,
    targets_df,
    preds_df,
    model_dir,
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

        # move to reset position
        robot.move_joints(joint_angles)

    # save results
    preds_df.to_csv(os.path.join(model_dir, 'predictions.csv'), index=False)
    targets_df.to_csv(os.path.join(model_dir, 'targets.csv'), index=False)
    error_plotter.final_plot(preds_df, targets_df)

    # finish 50mm above workframe origin then zero last joint 
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.move_joints((*robot.joint_angles[:-1], 0))
    robot.close()


if __name__ == "__main__":

    robot_str, sensor_str, tasks, models, _, device = setup_parse_args(
        robot='cr', 
        sensor='tactip_331',
        tasks=['edge_5d'],
        models=['simple_cnn'],
        device='cuda'
    )

    model_version = ''
    num_poses = 100

    # test the trained networks
    for task, model_str in zip(tasks, models):

        #  setup save dir
        save_dir = os.path.join(BASE_RUNS_PATH, robot_str+'_'+sensor_str, task, model_str+model_version)
        image_dir = os.path.join(save_dir, "processed_images")
        make_dir(save_dir)
        make_dir(image_dir)

        # set data and model dir
        model_dir = os.path.join(BASE_MODEL_PATH, robot_str+'_'+sensor_str, task, model_str+model_version)

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
        label_encoder = LabelEncoder(task_params, device)
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
            device=device
        )
        model.eval()

        pose_model = LabelledModel(
            model,
            preproc_params['image_processing'],
            label_encoder,
            device
        )

        test_model(
            robot,
            sensor,
            pose_model,
            collect_params,
            targets_df,
            preds_df,
            model_dir,
            error_plotter
        )
