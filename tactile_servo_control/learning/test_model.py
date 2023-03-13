# -*- coding: utf-8 -*-
"""
python test_model.py -r Sim -m simple_cnn -t edge_2d
"""
import os
import numpy as np

from tactile_learning.supervised.models import create_model
from tactile_learning.utils.utils_learning import load_json_obj

from utils_learning import PoseEncoder
from tactile_servo_control.collect_data.utils_collect_data import setup_parse
from tactile_servo_control import BASE_DATA_PATH, BASE_MODEL_PATH

from tactile_servo_control.servo_control.utils_servo_control import PoseModel
from tactile_servo_control.utils.setup_embodiment import setup_embodiment

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model_version = ''


def test_model(
    robot,
    sensor,
    model
):
    # start 50mm above workframe origin
    robot.move_linear((0, 0, 50, 0, 0, 0))

    # drop 10mm to contact object
    tap = (0, 0, -10, 0, 0, 0)

    # ==== data collection loop ====
    # for v in np.arange(-10,10,1):
    #     pose = np.array((v, 0, -3.5, 0, 0, 0))

    for v in np.arange(-180,180,10):
        pose = np.array((0, 0, -3.5, 0, 0, v))

        # move to above new pose (avoid changing pose in contact with object)
        robot.move_linear(pose - tap)

        # move to target positon 
        robot.move_linear(pose)

        # collect and process tactile image
        tactile_image = sensor.process()
        model.predict(tactile_image)

        # move back to above target positon 
        robot.move_linear(pose - tap)

        # report
        with np.printoptions(precision=1, suppress=True):
            print(f"\nPose {pose}")

    # finish 50mm above workframe origin then zero last joint 
    robot.move_linear((0, 0, 50, 0, 0, 0))
    robot.move_joints((*robot.joint_angles[:-1], 0))
    robot.close()


if __name__ == "__main__":
    pass

    input_args = {
        'tasks':  [['edge_2d'],    "['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d']"],
        'models': [['simple_cnn'], "['simple_cnn', 'posenet_cnn', 'nature_cnn', 'resnet', 'vit']"],
        'robot':  ['Sim',           "['Sim', 'MG400', 'CR']"],
        'device': ['cuda',         "['cpu', 'cuda']"],
    }
    tasks, model_types, robot, device = setup_parse(input_args)

    # test the trained networks
    for model_type in model_types:
        for task in tasks:

            # set data and model dir
            data_dir = os.path.join(BASE_DATA_PATH, robot, task, 'data')
            model_dir = os.path.join(BASE_MODEL_PATH, robot, task, model_type + model_version)

            # setup parameters
            env_params = load_json_obj(os.path.join(data_dir, 'env_params'))
            task_params = load_json_obj(os.path.join(model_dir, 'task_params'))
            model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
            sensor_params = load_json_obj(os.path.join(model_dir, 'sensor_params'))

            # create the label encoder/decoder
            label_encoder = PoseEncoder(**task_params, device=device)

            # setup embodiment, network and model
            robot, sensor = setup_embodiment(
                env_params, 
                sensor_params
            )

            # create the model
            model = create_model(
                in_dim=sensor_params['image_processing']['dims'],
                in_channels=1,
                out_dim=label_encoder.out_dim,
                model_params=model_params,
                saved_model_dir=model_dir,
                device=device
            )
            model.eval()

            pose_model = PoseModel(
                model,
                sensor_params['image_processing'],
                label_encoder,
                device=device
            )

            test_model(
                robot,
                sensor,
                pose_model
            )
