"""
python launch_collect_data.py -r sim -s tactip -t surface_3d
"""
import os

from tactile_data.tactile_servo_control import BASE_DATA_PATH
from tactile_data.utils_data import make_dir
from tactile_servo_control.utils.setup_embodiment import setup_embodiment
from tactile_servo_control.utils.setup_parse_args import setup_parse_args

from collect_data import collect_data
from setup_collect_data import setup_collect_data
from utils_collect_data import setup_target_df


def launch(
    robot='sim', 
    sensor='tactip',
    tasks=['edge_5d']
):
    data_params = {
        'data': 5000,
    }

    robot_str, sensor_str, tasks, _, _, _ = setup_parse_args(robot, sensor, tasks)

    for task in tasks:
        for dir_name, num_samples in data_params.items():

            # setup save dir
            save_dir = os.path.join(BASE_DATA_PATH, robot_str+'_'+sensor_str, task, dir_name)
            image_dir = os.path.join(save_dir, "images")
            make_dir(save_dir)
            make_dir(image_dir)

            # setup parameters
            collect_params, env_params, sensor_params = setup_collect_data(
                robot_str, 
                sensor_str,
                task, 
                save_dir
            )

            # setup embodiment
            robot, sensor = setup_embodiment(
                env_params, 
                sensor_params
            )

            # setup targets to collect
            target_df = setup_target_df(
                collect_params,
                num_samples, 
                save_dir
            )

            # collect          
            collect_data(
                robot,
                sensor, 
                target_df, 
                image_dir,
                collect_params
            )


if __name__ == "__main__":
    launch()
