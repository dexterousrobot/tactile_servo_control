# -*- coding: utf-8 -*-
"""
python launch_servo_control.py -r Sim -t edge_2d -s circle
"""
import os

from tactile_learning.utils.utils_learning import load_json_obj, make_dir

from tactile_servo_control import BASE_MODEL_PATH, BASE_TEST_PATH
from tactile_servo_control.utils.setup_embodiment import setup_embodiment
from tactile_learning.supervised.models import create_model
from tactile_servo_control.collect_data.utils_collect_data import setup_parse

from tactile_servo_control.learning.utils_learning import PoseEncoder

from servo_control import servo_control
from setup_servo_control import setup_control_params, setup_env_params 
from utils_servo_control import PoseModel

model_version = ''
test_version = ''


def launch():

    input_args = {
        'tasks':   [['edge_2d'], "['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d']"],
        'stimuli': [['circle'],  "['circle', 'square', 'clover', 'foil', 'saddle', 'bowl']"],
        'robot':   ['Sim',        "['Sim', 'MG400', 'CR']"],
        'device':  ['cuda',      "['cpu', 'cuda']"],
    }
    tasks, stimuli, robot, device = setup_parse(input_args)

    for task in tasks:
        for stimulus in stimuli:

            # setup save dir
            save_dir = os.path.join(BASE_TEST_PATH, robot, task, stimulus + test_version)
            image_dir = os.path.join(save_dir, "images")
            make_dir(save_dir)
            make_dir(image_dir)

            # setup environment, control, embodiment, network and model
            control_params = setup_control_params(task, save_dir)
            env_params = setup_env_params(robot, task, stimulus, save_dir)

            # set saved model dir
            model_dir = os.path.join(BASE_MODEL_PATH, robot, task, env_params['model_type'] + model_version)

            # load model and sensor params
            task_params = load_json_obj(os.path.join(model_dir, 'task_params'))
            model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
            sensor_params = load_json_obj(os.path.join(model_dir, 'sensor_params'))

            # create the label encoder/decoder
            label_encoder = PoseEncoder(**task_params, device=device)

            # setup embodiment, network and model
            embodiment = setup_embodiment(
                env_params, 
                sensor_params
            )

            model = create_model(
                in_dim=sensor_params['image_processing']['dims'],
                in_channels=1,
                out_dim=label_encoder.out_dim,
                model_params=model_params,
                device=device
            )
            model.eval()

            pose_model = PoseModel(
                model,
                sensor_params['image_processing'],
                label_encoder,
                device=device
            )

            # run the servo control
            servo_control(
                embodiment,
                pose_model,
                image_dir,
                **control_params
            )


if __name__ == "__main__":
    launch()