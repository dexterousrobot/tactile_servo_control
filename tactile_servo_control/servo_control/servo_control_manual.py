"""
python servo_control.py -t surface_3d
python servo_control.py -t edge_2d
python servo_control.py -t edge_3d
python servo_control.py -t edge_5d
python servo_control.py -t surface_3d edge_2d edge_3d edge_5d
"""

import os
import argparse
import numpy as np
import imageio

from tactile_gym.utils.general_utils import load_json_obj

from tactile_gym_servo_control.utils_robot_sim.setup_embodiment_env import setup_embodiment_env
from tactile_gym_servo_control.learning.setup_learning import setup_task

from tactile_gym_servo_control.servo_control.setup_sim_servo_control import setup_servo_control
from tactile_gym_servo_control.servo_control.utils_servo_control import Slider
from tactile_gym_servo_control.servo_control.utils_servo_control import keyboard
from tactile_gym_servo_control.servo_control.utils_plots import PlotContour3D as PlotContour
from tactile_gym_servo_control.utils.pose_transforms import transform_pose, inv_transform_pose

np.set_printoptions(precision=1, suppress=True)

model_path = os.path.join(os.path.dirname(__file__), "../../example_models/sim/simple_cnn")
videos_path = os.path.join(os.path.dirname(__file__), "../../example_videos")


def run_servo_control(
            embodiment, #model,
            ep_len=10000,
            ref_pose=[0, 0, 0, 0, 0, 0],
            p_gains=[0, 0, 0, 0, 0, 0],
            i_gains=[0, 0, 0, 0, 0, 0],
            i_clip=[-np.inf, np.inf],
            record_vid=False
        ):

    if record_vid:
        render_frames = []

    if embodiment.show_gui:
        slider = Slider(embodiment.slider, ref_pose)

    plotContour = PlotContour(embodiment.workframe)#, embodiment.stim_name)

    # initialise pose and integral term
    pose = [0, 0, 0, 0, 0, 0]
    int_delta = [0, 0, 0, 0, 0, 0]

    # move to initial pose from above workframe
    hover = embodiment.hover
    embodiment.move_linear(pose + hover)
    embodiment.move_linear(pose)

    # iterate through servo control
    for i in range(ep_len):

        # get current tactile observation
        tactile_image = embodiment.sensor_process()

        # get current TCP pose
        if embodiment.sim:
            tcp_pose = embodiment.get_tcp_pose()
        else:
            tcp_pose = pose

        # control robot with key press
        delta = keyboard(embodiment)
        if delta is None: 
            break # quit

        # apply pi(d) control to reduce delta
        int_delta = delta + 0.9 * np.array(int_delta)
        int_delta = np.clip(int_delta, *i_clip)

        output = p_gains * delta  +  i_gains * int_delta 
        
        # new pose combines output pose with tcp_pose 
        pose = inv_transform_pose(output, tcp_pose)

        # move to new pose
        embodiment.move_linear(pose)

        # slider control
        if embodiment.show_gui:
            ref_pose = slider.slide(ref_pose)
           
        # show tcp if sim
        if embodiment.show_gui and embodiment.sim:
            embodiment.arm.draw_TCP(lifetime=10.0)
        
        # render frames,
        if record_vid:
            render_img = embodiment.render()
            render_frames.append(render_img)

        # report
        print(f'\nstep {i+1}: pose: {pose}', end='')
        plotContour.update(pose)

    # move to above final pose
    embodiment.move_linear(pose + hover)
    embodiment.close()

    if record_vid:
        imageio.mimwrite(
            os.path.join(videos_path, "render.mp4"),
            np.stack(render_frames),
            fps=24
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose task from ['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d'].",
        default=['edge_5d']
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        help="Choose device from ['cpu', 'cuda'].",
        default='cuda'
    )

    # parse arguments
    args = parser.parse_args()
    tasks = args.tasks
    device = args.device
    version = ''

    for task in tasks:

        # setup servo control for the task
        env_params_list, control_params = setup_servo_control[task]()

        # set save dir
        task += version
        model_dir = os.path.join(model_path, task)

        # load params
        sensor_params = load_json_obj(os.path.join(model_dir, 'sensor_params'))

        # perform the servo control
        for env_params in env_params_list:

            embodiment = setup_embodiment_env(
                **env_params,
                sensor_params=sensor_params, quick_mode=True
            )

            run_servo_control(
                embodiment,
                **control_params
            )
