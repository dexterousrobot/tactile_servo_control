"""
python demo_marker_generation.py -r abb -s tactip -t edge_2d
"""
import os
import itertools as it

from tactile_data.tactile_servo_control import BASE_DATA_PATH
from tactile_image_processing.utils import load_json_obj
from tactile_learning.supervised.marker_generator import demo_marker_generation

from tactile_servo_control.learning.setup_training import setup_learning, csv_row_to_label
from tactile_servo_control.utils.parse_args import parse_args


if __name__ == '__main__':

    args = parse_args(
        robot='abb',
        sensor='tactip',
        tasks=['edge_2d'],
        data_dirs=['train', 'val']
    )

    output_dir = '_'.join([args.robot, args.sensor])

    data_dirs = [
        os.path.join(BASE_DATA_PATH, output_dir, *i) for i in it.product(args.tasks, args.data_dirs)
    ]

    learning_params = setup_learning()
    marker_params = load_json_obj(os.path.join(data_dirs[0], 'processed_marker_params'))

    demo_marker_generation(
        data_dirs,
        csv_row_to_label,
        learning_params,
        num_markers=marker_params['num_markers']
    )
