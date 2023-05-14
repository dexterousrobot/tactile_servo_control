"""
python replay_servo_control.py -r sim -s tactip -t edge_2d -o circle
"""
import os
import itertools as it

from tactile_data.tactile_servo_control import BASE_MODEL_PATH, BASE_RUNS_PATH
from tactile_image_processing.utils import load_json_obj
from tactile_learning.supervised.models import create_model

from tactile_servo_control.utils.label_encoder import LabelEncoder
from tactile_servo_control.utils.labelled_model import LabelledModel
from tactile_servo_control.servo_control.launch_servo_control import servo_control
from tactile_servo_control.utils.controller import PIDController
from tactile_servo_control.utils.parse_args import parse_args
from tactile_servo_control.utils.setup_embodiment import setup_embodiment


def replay(args):

    output_dir = '_'.join([args.robot, args.sensor])

    for args.task, args.model, args.object in it.product(args.tasks, args.models, args.objects):

        model_dir_name = '_'.join(filter(None, [args.model, *args.model_version]))
        run_dir_name = '_'.join(filter(None, [args.object, *args.run_version]))

        # setup save dir
        run_dir = os.path.join(BASE_RUNS_PATH, output_dir, args.task, run_dir_name)
        image_dir = os.path.join(run_dir, "processed_images")
        control_params = load_json_obj(os.path.join(run_dir, 'control_params'))
        env_params = load_json_obj(os.path.join(run_dir, 'env_params'))
        task_params = load_json_obj(os.path.join(run_dir, 'task_params'))

        # load model, task and preproc parameters
        model_dir = os.path.join(BASE_MODEL_PATH, output_dir, args.task, model_dir_name)
        model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
        model_label_params = load_json_obj(os.path.join(model_dir, 'model_label_params'))
        model_image_params = load_json_obj(os.path.join(model_dir, 'model_image_params'))
        sensor_params = {'type': 'replay'}
        # env_params['work_frame'] += np.array([0, 0, 2, 0, 0, 0])

        # setup the robot and sensor
        robot, sensor = setup_embodiment(
            env_params,
            sensor_params
        )

        # setup the controller
        pid_controller = PIDController(**control_params)

        # create the label encoder/decoder
        label_encoder = LabelEncoder(model_label_params, device=args.device)

        # setup the model
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
            device=args.device
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

    args = parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['edge_2d'],
        models=['simple_cnn'],
        model_version=[''],
        objects=['circle', 'square'],
        run_version=[''],
        device='cuda'
    )

    replay(args)
