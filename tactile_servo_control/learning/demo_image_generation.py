"""
python demo_image_generation.py -r sim -s tactip -t edge_2d
"""
import os

from tactile_data.tactile_servo_control import BASE_DATA_PATH, BASE_RUNS_PATH
from tactile_learning.supervised.image_generator import demo_image_generation
from tactile_servo_control.utils.setup_parse_args import setup_parse_args

from setup_training import setup_learning, csv_row_to_label


if __name__ == '__main__':

    robot_str, sensor_str, tasks, _, _, _ = setup_parse_args(
        robot='sim', 
        sensor='tactip',
        tasks=['edge_5d'],
    )

    data_dirs = [
        # *[os.path.join(BASE_DATA_PATH, robot_str+'_'+sensor_str, task, 'train') for task in tasks],
        # *[os.path.join(BASE_DATA_PATH, robot_str+'_'+sensor_str, task, 'val') for task in tasks]
        *[os.path.join(BASE_RUNS_PATH, robot_str+'_'+sensor_str, task, 'simple_cnn') for task in tasks],
    ]

    learning_params, preproc_params = setup_learning()

    demo_image_generation(
        data_dirs,
        csv_row_to_label,
        learning_params,
        preproc_params['image_processing'],
        {} # sensor_params['augmentation']
    )
