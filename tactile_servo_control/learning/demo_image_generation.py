"""
python demo_image_generation.py -r sim -s tactip -t edge_2d
"""
import os

from tactile_data.tactile_servo_control import BASE_DATA_PATH
from tactile_learning.supervised.image_generator import demo_image_generation
from tactile_servo_control.utils.setup_parse_args import setup_parse_args

from utils_learning import LabelEncoder
from setup_training import setup_learning


if __name__ == '__main__':

    robot_str, sensor_str, tasks, _, _, _ = setup_parse_args(
        robot='sim', 
        sensor='tactip',
        tasks=['edge_5d'],
    )

    data_dirs = [
        *[os.path.join(BASE_DATA_PATH, robot_str+'_'+sensor_str, task_str, 'train') for task_str in tasks],
        # *[os.path.join(BASE_DATA_PATH, robot+'_'+sensor, task, 'val') for task in tasks]
    ]

    learning_params, preproc_params = setup_learning()

    demo_image_generation(
        data_dirs,
        LabelEncoder.csv_row_to_label,
        learning_params,
        preproc_params['image_processing'],
        {} # sensor_params['augmentation']
    )
