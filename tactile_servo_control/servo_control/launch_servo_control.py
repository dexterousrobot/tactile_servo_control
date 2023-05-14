"""
python launch_servo_control.py -r sim -s tactip -t edge_2d -o circle square
"""
import os
import itertools as it
import time as t
import numpy as np

from cri.transforms import inv_transform_euler
from tactile_data.tactile_servo_control import BASE_MODEL_PATH, BASE_RUNS_PATH
from tactile_image_processing.utils import load_json_obj, make_dir
from tactile_learning.supervised.models import create_model
from user_input.slider import Slider

from tactile_servo_control.servo_control.setup_servo_control import setup_servo_control
from tactile_servo_control.utils.label_encoder import LabelEncoder
from tactile_servo_control.utils.labelled_model import LabelledModel
from tactile_servo_control.utils.controller import PIDController
from tactile_servo_control.utils.parse_args import parse_args
from tactile_servo_control.utils.setup_embodiment import setup_embodiment
from tactile_servo_control.utils.utils_plots import PlotContour3D as PlotContour


def servo_control(
    robot,
    sensor,
    pose_model,
    controller,
    image_dir,
    task_params,
    show_plot=True,
    show_slider=False,
):

    # initialize peripherals
    if show_plot:
        plotContour = PlotContour(robot.coord_frame)
    if show_slider:
        slider = Slider(controller.ref)

    # move to initial pose from 50mm above workframe
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.move_joints([*robot.joint_angles[:-1], 0])

    # zero pose and clock
    pose = [0, 0, 0, 0, 0, 0]
    robot.move_linear(pose)

    # turn on servo mode if set
    robot.controller.servo_mode = task_params.get('servo_mode', False)
    robot.controller.time_delay = task_params.get('time_delay', 0.0)

    # timed iteration through servo control
    t_0 = t.time()
    for i in range(task_params['num_iterations']):

        # get current tactile observation
        image_outfile = os.path.join(image_dir, f'image_{i}.png')
        tactile_image = sensor.process(image_outfile)

        # predict pose from observations
        pred_pose = pose_model.predict(tactile_image)

        # servo control output in sensor frame
        servo = controller.update(pred_pose)

        # new pose applies servo in end effector frame
        pose = inv_transform_euler(servo, robot.pose)
        robot.move_linear(pose)

        # optional peripheral: plot trajectory, reference slider
        if show_plot:
            plotContour.update(pose)
        if show_slider:
            controller.ref = slider.read()

        # report
        with np.printoptions(precision=1, suppress=True):
            print(f'\n step {i+1} time {np.array([t.time()-t_0])}: pose: {pose}')

    # finish 50mm above initial pose and zero joint_6
    robot.controller.servo_mode = False
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.move_joints((*robot.joint_angles[:-1], 0))
    robot.close()

    # optionally save plot and render view
    if show_plot:
        plot_outfile = os.path.join(image_dir, r"../trajectory.png")
        plotContour.save(plot_outfile)


def launch(args):

    output_dir = '_'.join([args.robot, args.sensor])

    for args.task, args.model in it.product(args.tasks, args.models):
        for args.object, args.sample_num in zip(args.objects, args.sample_nums):

            model_dir_name = '_'.join(filter(None, [args.model, *args.model_version]))
            run_dir_name = '_'.join(filter(None, [args.object, *args.run_version]))

            # setup save dir
            save_dir = os.path.join(BASE_RUNS_PATH, output_dir, args.task, run_dir_name)
            image_dir = os.path.join(save_dir, "processed_images")
            make_dir(save_dir)
            make_dir(image_dir)

            # load model, environment and image processing parameters
            model_dir = os.path.join(BASE_MODEL_PATH, output_dir, args.task, model_dir_name)
            env_params = load_json_obj(os.path.join(model_dir, 'env_params'))
            model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
            model_image_params = load_json_obj(os.path.join(model_dir, 'model_image_params'))
            model_label_params = load_json_obj(os.path.join(model_dir, 'model_label_params'))
            if os.path.isfile(os.path.join(model_dir, 'processed_image_params.json')):
                sensor_image_params = load_json_obj(os.path.join(model_dir, 'processed_image_params'))
            else:
                sensor_image_params = load_json_obj(os.path.join(model_dir, 'sensor_image_params'))

            # setup control and update env parameters from data_dir
            control_params, env_params, task_params = setup_servo_control(
                args.sample_num,
                args.task,
                args.object,
                model_dir_name,
                env_params,
                save_dir
            )

            # setup the robot and sensor
            robot, sensor = setup_embodiment(
                env_params,
                sensor_image_params
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
        sample_nums=[100, 100],
        run_version=[''],
        device='cuda'
    )

    launch(args)
