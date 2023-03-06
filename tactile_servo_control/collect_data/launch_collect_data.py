# -*- coding: utf-8 -*-
"""
python launch_collect_data.py -r Sim -t edge_2d
"""
import os

from tactile_servo_control import BASE_DATA_PATH
from tactile_servo_control.utils.setup_embodiment import setup_embodiment
from tactile_learning.utils.utils_learning import make_dir

from collect_data import collect_data
from setup_collect_data import setup_collect_data
from utils_collect_data import setup_target_df, setup_parse

collect_params = {
    'data': 10,#5000,
    # 'train': 4000,
    # 'val': 1000
}

def launch():

    tasks, robot = setup_parse({
        'tasks':  [['edge_2d'],   "['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d']"],
        'robot':  ['MG400',          "['Sim', 'MG400', 'CR']"],
    })
    
    for task in tasks:
        for dir_name, num_samples in collect_params.items():

            # setup save dir
            save_dir = os.path.join(BASE_DATA_PATH, robot, task, dir_name)
            image_dir = os.path.join(save_dir, "images")
            make_dir(save_dir)
            make_dir(image_dir)

            # setup parameters
            sensor_params, pose_params, env_params = setup_collect_data(
                robot, 
                task, 
                save_dir
            )

            # setup embodiment
            embodiment = setup_embodiment(
                env_params, 
                sensor_params
            )

            # setup targets to collect
            target_df = setup_target_df(
                num_samples, 
                save_dir, 
                **pose_params
            )

            # collect          
            collect_data(
                embodiment, 
                target_df, 
                image_dir
            )


if __name__ == "__main__":
    launch()
