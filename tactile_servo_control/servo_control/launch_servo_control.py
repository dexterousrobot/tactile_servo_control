# -*- coding: utf-8 -*-
"""
python launch_servo_control.py -t surface_3d edge_2d edge_3d edge_5d
"""
import os

from tactile_learning.utils.utils_learning import load_json_obj, make_dir

from tactile_servo_control import BASE_MODEL_PATH, BASE_TEST_PATH
from tactile_servo_control.utils.setup_embodiment import setup_embodiment
from tactile_learning.supervised.models import create_model
from tactile_servo_control.collect_data.utils_collect_data import setup_parse

from servo_control import servo_control
from setup_servo_control import setup_servo_control 
from utils_servo_control import PoseModel

model_version = ''#_CR'
test_version = ''


def launch():

    input_args = {
        'tasks':   [['edge_2d'], "['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d']"],
        'stimuli': [['circle'],  "['circle', 'square', 'clover', 'foil', 'saddle', 'bowl']"],
        'reality': ['sim',       "['sim', 'real'"],
        'device':  ['cuda',      "['cpu', 'cuda']"],
    }
    tasks, stimuli, reality, device = setup_parse(input_args)

    for task in tasks:
        for stimulus in stimuli:

            # setup save dir
            save_dir = os.path.join(BASE_TEST_PATH, reality, task, stimulus + test_version)
            image_dir = os.path.join(save_dir, "images")
            make_dir(save_dir)
            make_dir(image_dir)

            # setup environment, control, embodiment, network and model
            env_params, control_params = setup_servo_control(
                reality, 
                task, 
                stimulus,
                save_dir
            )

            # set saved model dir
            model_dir = os.path.join(BASE_MODEL_PATH, reality, task, env_params['model_type'] + model_version)

            # load model and sensor params
            network_params = load_json_obj(os.path.join(model_dir, 'model_params'))
            image_processing_params = load_json_obj(os.path.join(model_dir, 'image_processing_params'))
            sensor_params = load_json_obj(os.path.join(model_dir, 'sensor_params'))
            task_params = load_json_obj(os.path.join(model_dir, 'task_params'))
            pose_params = load_json_obj(os.path.join(model_dir, 'pose_params'))

            # setup embodiment, network and model
            embodiment = setup_embodiment(
                reality,
                env_params, 
                sensor_params
            )

            model = create_model(
                image_processing_params['dims'],
                task_params['out_dim'],
                network_params,
                saved_model_dir=model_dir,
                device=device
            )
            model.eval()

            pose_model = PoseModel(
                model,
                image_processing_params,
                pose_params,
                task_params['label_names'],
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