"""
python demo_image_generation.py -r sim -s tactip -t edge_2d
"""
import os
import itertools as it

from tactile_data.tactile_servo_control import BASE_DATA_PATH
from tactile_learning.supervised.image_generator import demo_image_generation

from tactile_servo_control.learning.setup_training import setup_learning, setup_model_image
from tactile_servo_control.learning.setup_training import csv_row_to_label
from tactile_servo_control.utils.parse_args import parse_args

if __name__ == '__main__':

    args = parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['edge_2d'],
        data_dirs=['train_data', 'val_data']
    )

    output_dir = '_'.join([args.robot, args.sensor])

    data_dirs = [
        os.path.join(BASE_DATA_PATH, output_dir, *i) for i in it.product(args.tasks, args.data_dirs)
    ]

    learning_params = setup_learning()
    image_params = setup_model_image()

    demo_image_generation(
        data_dirs,
        csv_row_to_label,
        learning_params,
        image_params['image_processing'],
        image_params['augmentation'],
    )
