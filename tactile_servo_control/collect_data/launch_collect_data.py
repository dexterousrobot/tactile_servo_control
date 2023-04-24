"""
python launch_collect_data.py -r sim -s tactip -t edge_2d
"""
import os

from tactile_data.tactile_servo_control import BASE_DATA_PATH
from tactile_data.collect_data.collect_data import collect_data
from tactile_data.collect_data.process_image_data import process_image_data, partition_data
from tactile_data.collect_data.setup_targets import setup_targets
from tactile_data.utils import make_dir

from tactile_servo_control.collect_data.setup_collect_data import setup_collect_data
from tactile_servo_control.utils.parse_args import parse_args
from tactile_servo_control.utils.setup_embodiment import setup_embodiment


def launch(args):

    output_dir = '_'.join([args.robot, args.sensor])

    for args.task in args.tasks:
        for args.data_dir, args.sample_num in zip(args.data_dirs, args.sample_nums):

            # setup save dir
            save_dir = os.path.join(BASE_DATA_PATH, output_dir, args.task, args.data_dir)
            image_dir = os.path.join(save_dir, "sensor_images")
            make_dir(save_dir)
            make_dir(image_dir)

            # setup parameters
            collect_params, env_params, sensor_image_params = setup_collect_data(
                args.robot,
                args.sensor,
                args.task,
                save_dir
            )

            # setup embodiment
            robot, sensor = setup_embodiment(
                env_params,
                sensor_image_params
            )

            # setup targets to collect
            target_df = setup_targets(
                collect_params,
                args.sample_num,
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


def process_images(args, image_params, split=None):

    output_dir = '_'.join([args.robot, args.sensor])

    for args.task in args.tasks:
        path = os.path.join(BASE_DATA_PATH, output_dir, args.task)
        data_dirs = partition_data(path, args.data_dirs, split)
        process_image_data(path, data_dirs, image_params)


if __name__ == "__main__":

    args = parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['edge_2d'],
        data_dirs=['data'],
        sample_nums=[5000]
    )

    image_params = {
        # "thresh": [61, 5],
        # "circle_mask_radius": 210, # 140 ABB tactip # 210 CR midi
        # sim (12, 12, 240, 240) # CR midi (5, 10, 425, 430) # MG400 mini (10, 10, 310, 310) # ABB tactip (25, 25, 305, 305)
        "bbox": (12, 12, 240, 240)
    }

    launch(args)
    process_images(args, image_params, split=0.8)
