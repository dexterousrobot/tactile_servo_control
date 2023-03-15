# -*- coding: utf-8 -*-
import os
import time as t
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
            show_plot=True,
            show_slider=False,
            servo_mode=False
        ):

    # initialize peripherals
    if show_plot:
        plotContour = PlotContour(robot.coord_frame)
    if show_slider:
        slider = Slider(controller.ref)

    # move to initial pose from 50mm above workframe
    robot.move_joints([*robot.joint_angles[:-1], 0])
    robot.move_linear((0, 0, -50, 0, 0, 0))

    # zero pose and clock
    pose = [0, 0, 0, 0, 0, 0]
    robot.move_linear(pose)
    t_0 = t.time()

    # turn on servo mode if set
    robot.controller.servo_mode = servo_mode 

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


if __name__ == '__main__':
    pass