"""
python demo_image_generation.py -r sim -s tactip -t edge_2d
"""
import os

from tactile_data.tactile_servo_control import BASE_DATA_PATH
from tactile_learning.supervised.image_generator import demo_image_generation

from tactile_servo_control.learning.setup_training import setup_learning, csv_row_to_label
from tactile_servo_control.utils.parse_args import parse_args


if __name__ == '__main__':

    args = parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['edge_2d'],
        version=['']
    )

    output_dir = '_'.join([args.robot, args.sensor])
    train_dir_name = '_'.join(filter(None, ["train", *args.version]))
    val_dir_name = '_'.join(filter(None, ["val", *args.version]))

    data_dirs = [
        *[os.path.join(BASE_DATA_PATH, output_dir, task, train_dir_name) for task in args.tasks],
        *[os.path.join(BASE_DATA_PATH, output_dir, task, val_dir_name) for task in args.tasks]
    ]

    learning_params, preproc_params = setup_learning()

    demo_image_generation(
        data_dirs,
        csv_row_to_label,
        learning_params,
        preproc_params['image_processing'],
        preproc_params['augmentation'],
    )
