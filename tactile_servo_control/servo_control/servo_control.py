# -*- coding: utf-8 -*-
import os
import numpy as np

from cri.transforms import inv_transform_euler, transform_euler
from user_input.slider import Slider
from utils_servo_control import PlotContour3D as PlotContour


def servo_control(
            robot, 
            sensor, 
            model,
            controller,
            image_dir,
            num_iterations=100,
            plot_trajectory=True
        ):

    # initialize peripherals
    slider = Slider(controller.ref)
    if plot_trajectory:
        plotContour = PlotContour(robot.coord_frame)

    # initialise controller and pose
    pose = [0, 0, 0, 0, 0, 0]

    # move to initial pose from 50mm above workframe
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.move_linear(pose)

    # iterate through servo control
    for i in range(num_iterations):

        # get current tactile observation
        image_outfile = os.path.join(image_dir, f'image_{i}.png')
        tactile_image = sensor.process(image_outfile)

        # predict pose from observations
        pred_pose = model.predict(tactile_image)

        # servo control output in sensor frame
        servo = controller.update(pred_pose)

        # new pose applies servo in end effector frame
        pose = inv_transform_euler(servo, robot.pose)
        robot.move_linear(pose)

        # optionally plot and render view
        if plot_trajectory:
            plotContour.update(pose)

        # use slider to set reference
        controller.ref = slider.read()

        # report
        with np.printoptions(precision=1, suppress=True):
            print(f'\n step {i+1}: pose: {pose}')

    # finish 50mm above initial pose and zero joint_6
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.move_joints((*robot.joint_angles[:-1], 0))
    robot.close()

    # optionally save plot and render view
    if plot_trajectory:
        plot_outfile = os.path.join(image_dir, r"../trajectory.png")
        plotContour.save(plot_outfile)


if __name__ == '__main__':
    pass