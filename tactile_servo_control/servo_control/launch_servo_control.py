"""
python launch_servo_control.py -r cr -s tactip_331 -t edge_5d -o circle
"""
import os

from tactile_data.tactile_servo_control import BASE_MODEL_PATH, BASE_RUNS_PATH
from tactile_data.utils_data import load_json_obj, make_dir
from tactile_learning.supervised.models import create_model
from tactile_servo_control.learning.utils_learning import LabelEncoder, LabelledModel
from tactile_servo_control.utils.controller import PIDController
from tactile_servo_control.utils.setup_embodiment import setup_embodiment
from tactile_servo_control.utils.setup_parse_args import setup_parse_args

from servo_control import servo_control
from setup_servo_control import setup_servo_control


def launch():

    robot_str, sensor_str, tasks, models, objects, device = setup_parse_args(
        robot='sim', 
        sensor='tactip', 
        tasks=['edge_2d'],
        models=['simple_cnn'],
        objects=['circle'],
        device='cuda'
    )

    run_version = ''

    for task, model_str, object_str in zip(tasks, models, objects):
        
        # setup save dir
        save_dir = os.path.join(BASE_RUNS_PATH, robot_str+'_'+sensor_str, task, object_str + run_version)
        image_dir = os.path.join(save_dir, "processed_images")
        make_dir(save_dir)
        make_dir(image_dir)

        # load model, task and preproc parameters
        model_dir = os.path.join(BASE_MODEL_PATH, robot_str+'_'+sensor_str, task, model_str)
        model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
        preproc_params = load_json_obj(os.path.join(model_dir, 'preproc_params'))
        sensor_params = load_json_obj(os.path.join(model_dir, 'sensor_params'))

        # setup control and update env parameters from data_dir
        control_params, env_params, task_params = setup_servo_control(
            task, 
            object_str, 
            model_str, 
            model_dir,
            save_dir
        )
        
        # setup the robot and sensor 
        robot, sensor = setup_embodiment(
            env_params,     
            sensor_params
        )

        # setup the controller
        pid_controller = PIDController(**control_params)

        # create the label encoder/decoder
        label_encoder = LabelEncoder(task_params, device=device)

        # setup the model
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
            device=device
        )

        # run the servo control
        servo_control(
            robot,
            sensor,
            pose_model,
            pid_controller,
            image_dir,
            task_params
        )


if __name__ == "__main__":
    launch()
