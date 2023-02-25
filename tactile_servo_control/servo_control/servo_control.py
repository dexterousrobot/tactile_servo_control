# -*- coding: utf-8 -*-
import os
import imageio
import numpy as np

from cri.transforms import inv_transform_pose
from tactile_servo_control.utils.controller import PIDController
from user_input.slider import Slider
from utils_servo_control import PlotContour3D as PlotContour


def servo_control(
            embodiment, 
            pose_model,
            image_dir,
            ep_len=100,
            pid_params={},
            ref_pose=[0, 0, 0, 0, 0, 0],
            record_vid=False,
            plot_trajectory=False
        ):

    # initialize peripherals
    slider = Slider(ref_pose)
    if plot_trajectory:
        plotContour = PlotContour(embodiment.coord_frame)
    if record_vid:
        render_frames = []

    # initialise controller and pose
    controller = PIDController(**pid_params)
    pose = [0, 0, 0, 0, 0, 0]

    # move to initial pose from above workframe
    hover = embodiment.hover
    embodiment.move_linear(pose + hover)
    embodiment.move_linear(pose)

    # iterate through servo control
    for i in range(ep_len):

        # get current tactile observation
        image_outfile = os.path.join(image_dir, f'image_{i}.png')
        tactile_image = embodiment.sensor.process(image_outfile)

        # predict pose from observations
        pred_pose = pose_model.predict(tactile_image)

        # aervo control output in sensor frame
        servo = controller.update(pred_pose, ref_pose)
        
        # new pose applies servo to tcp_pose 
        tcp_pose = embodiment.pose
        pose = inv_transform_pose(servo, tcp_pose)

        # move to new pose
        embodiment.move_linear(pose)

        # slider control
        ref_pose = slider.read(ref_pose)
       
        # optionally plot and render view
        if plot_trajectory:
            plotContour.update(pose)
        if record_vid:
            render_img = embodiment.controller._client._sim_env.render()
            render_frames.append(render_img)

        # report
        with np.printoptions(precision=1, suppress=True):
            print(f'\n step {i+1}: pose: {pose}')
        # embodiment.controller._client._sim_env.arm.draw_TCP(lifetime=10.0)

    # move to above final pose
    embodiment.move_linear(pose + hover)
    embodiment.close()

    # optionally save plot and render view
    if plot_trajectory:
        plot_outfile = os.path.join(image_dir, r"../trajectory.png")
        plotContour.save(plot_outfile)
    if record_vid:
        imageio.mimwrite(
            os.path.join(image_dir, r"../render.mp4"),
            np.stack(render_frames), fps=24
        )


if __name__ == '__main__':
    pass