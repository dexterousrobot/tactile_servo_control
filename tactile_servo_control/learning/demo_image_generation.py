import os

from setup_learning import setup_learning
from utils_learning import csv_row_to_label

from tactile_learning.supervised.image_generator import demo_image_generation

from tactile_servo_control.collect_data.utils_collect_data import setup_parse
from tactile_servo_control import BASE_DATA_PATH


if __name__ == '__main__':

    input_args = {
        'tasks':  [['edge_2d'],    "['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d']"],
        'robot':  ['CR',           "['Sim', 'MG400', 'CR']"]
    }
    tasks, robot = setup_parse(input_args)

    data_dirs = [
        *[os.path.join(BASE_DATA_PATH, robot, task, 'train') for task in tasks],
        *[os.path.join(BASE_DATA_PATH, robot, task, 'val') for task in tasks]
    ]

    learning_params, sensor_params = setup_learning(data_dirs)

    demo_image_generation(
        data_dirs,
        csv_row_to_label,
        learning_params,
        sensor_params['image_processing'],
        {} # sensor_params['augmentation']
    )
