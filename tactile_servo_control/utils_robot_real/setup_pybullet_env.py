import pybullet as p
import pybullet_utils.bullet_client as bc

from tactile_gym_servo_control.utils_robot_sim.robot_embodiment import RobotEmbodiment


def setup_pybullet_env(
    show_gui=True
):

    if show_gui:
        pb = bc.BulletClient(connection_mode=p.GUI)
    else:
        pb = bc.BulletClient(connection_mode=p.DIRECT)

    sensor_params = {
        "name": "tactip",
        "type": "standard",
        "core": "no_core",
        "dynamics": {},
        "image_size": [128, 128]
    }

    # create the robot and sensor embodiment
    embodiment = RobotEmbodiment(
        pb,
        workframe_pos=[0, 0, 0],
        workframe_rpy=[0, 0, 0],
        t_s_params=sensor_params,
        show_tactile=False,
    )

    return embodiment
