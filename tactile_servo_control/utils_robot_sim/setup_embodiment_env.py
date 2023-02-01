import os
import cv2
import numpy as np

from tactile_gym_servo_control.utils_robot_sim.setup_pybullet_env import setup_pybullet_env

stimuli_path = os.path.join(os.path.dirname(__file__), 'stimuli')


def setup_embodiment_env(
    workframe=[600, 0, 52.5, -180, 0, 90],
    stim_name="square",
    stim_pose=[600, 0, 12.5, 0, 0, 0],
    stim_scale=1,
    sensor_params = {},
    hover=[0, 0, -7.5, 0, 0, 0],
    show_gui=True, 
    show_tactile=True,
    quick_mode=False
):

    stim_path = os.path.join(
        stimuli_path, stim_name, stim_name + ".urdf"
    )

    # setup the robot
    embodiment = setup_pybullet_env(
        workframe,
        stim_path,
        stim_pose,
        stim_scale,
        sensor_params,
        show_gui,
        show_tactile,
        quick_mode
    )

    # setup the tactile sensor
    def sensor_process(outfile=None):
        img = embodiment.get_tactile_observation()
        if outfile is not None:
            cv2.imwrite(outfile, img)
        return img

    embodiment.sensor_process = sensor_process
    embodiment.slider = embodiment
    embodiment.sim = True

    embodiment.hover = np.array(hover)
    embodiment.show_gui = show_gui
    embodiment.workframe = workframe
    embodiment.stim_name = stim_name

    return embodiment


if __name__ == '__main__':
    pass
