# -*- coding: utf-8 -*-
"""
python demo_image_generation.py -t surface_3d edge_2d edge_3d edge_5d
"""

import os

from tactile_servo_control.learning.setup_learning import parse_args
from tactile_servo_control.learning.setup_learning import setup_learning
from tactile_servo_control.learning.utils_learning import csv_row_to_label

from tactile_learning.supervised.image_generator import demo_image_generation

from tactile_servo_control import BASE_DATA_PATH


if __name__ == '__main__':

    args = parse_args()
    tasks = args.tasks

    learning_params, image_processing_params, augmentation_params = setup_learning()

    data_dirs = [
        *[os.path.join(BASE_DATA_PATH, task, 'train') for task in tasks],
        *[os.path.join(BASE_DATA_PATH, task, 'val') for task in tasks]
    ]

    demo_image_generation(
        data_dirs,
        csv_row_to_label,
        learning_params,
        image_processing_params,
        augmentation_params
    )
