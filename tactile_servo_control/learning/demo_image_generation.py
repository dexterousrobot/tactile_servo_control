"""
python demo_image_generation.py -t surface_3d
python demo_image_generation.py -t edge_2d
python demo_image_generation.py -t edge_3d
python demo_image_generation.py -t edge_5d
python demo_image_generation.py -t surface_3d edge_2d edge_3d edge_5d
"""

import os
import argparse

from tactile_servo_control import BASE_DATA_PATH

from tactile_servo_control.learning.utils_learning import csv_row_to_label
from tactile_learning.supervised.image_generator import demo_image_generation

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose task from ['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d'].",
        default=['surface_3d']
    )
    args = parser.parse_args()
    tasks = args.tasks

    learning_params = {
        'batch_size':  8,
        'shuffle': True,
        'n_cpu': 1,
    }

    image_processing_params = {
        'dims': (128, 128),
        'bbox': None,
        'thresh': None,
        'stdiz': False,
        'normlz': True,
    }

    augmentation_params = {
        'rshift': None,  # (0.015, 0.015),
        'rzoom':   None,
        'brightlims': None,
        'noise_var': None,
    }

    data_dirs = [
        os.path.join(BASE_DATA_PATH, task, 'train') for task in tasks
        # os.path.join(BASE_DATA_PATH, task, 'val') for task in tasks
    ]

    demo_image_generation(
        data_dirs,
        csv_row_to_label,
        learning_params,
        image_processing_params,
        augmentation_params
    )
