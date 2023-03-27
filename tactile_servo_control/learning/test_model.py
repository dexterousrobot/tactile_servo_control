"""
python test_model.py -r sim -m simple_cnn -t edge_2d
"""
import os
import numpy as np
import pandas as pd

from tactile_data.tactile_servo_control import BASE_DATA_PATH, BASE_MODEL_PATH
from tactile_data.utils_data import load_json_obj
from tactile_learning.supervised.models import create_model
from tactile_servo_control.collect_data.utils_collect_data import setup_target_df
from tactile_servo_control.utils.setup_parse_args import setup_parse_args
from tactile_servo_control.utils.setup_embodiment import setup_embodiment

from utils_learning import LabelEncoder, LabelledModel
from utils_plots import ErrorPlotter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def test_model(
    robot,
    sensor,
    pose_model,
    task_params,
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
        pose = row.loc[task_params['pose_label_names']].values.astype(float)
        shear = row.loc[task_params['shear_label_names']].values.astype(float)

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
        tactile_image = sensor.process()
        preds_df.loc[i] = pose_model.predict(tactile_image)

        # move above the target pose
        robot.move_linear(pose - clearance)

        # if sorted, don't move to reset position
        if not task_params['sort']:
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
        tasks=['edge_3d'],
        models=['simple_cnn'],
        device='cuda'
    )

    model_version = '_sorted'
    num_poses = 50

    # test the trained networks
    for model_str, task in zip(models, tasks):

        # set data and model dir
        data_dir = os.path.join(BASE_DATA_PATH, robot_str+'_'+sensor_str, task, 'train')
        model_dir = os.path.join(BASE_MODEL_PATH, robot_str+'_'+sensor_str, task, model_str + model_version)

        # load model params
        model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
        preproc_params = load_json_obj(os.path.join(model_dir, 'preproc_params'))
        sensor_params = load_json_obj(os.path.join(model_dir, 'sensor_params'))
        task_params = load_json_obj(os.path.join(model_dir, 'task_params'))

        # create the label encoder/decoder
        label_encoder = LabelEncoder(task_params, device)
        task_params['target_label_names'] = task_params['pose_label_names']
        error_plotter = ErrorPlotter(task_params, model_dir, name='test_plot.png')

        # load data parameters
        env_params = load_json_obj(os.path.join(data_dir, 'env_params'))
        task_params = load_json_obj(os.path.join(data_dir, 'task_params'))
        targets_df = setup_target_df(task_params, num_poses) 
        preds_df = pd.DataFrame(columns=task_params['pose_label_names'])

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
            task_params,
            targets_df,
            preds_df,
            model_dir,
            error_plotter
        )
