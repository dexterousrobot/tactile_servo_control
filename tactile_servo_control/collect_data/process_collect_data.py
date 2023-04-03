"""
python process_collect_data.py -r cr -s tactip_331 -t edge_5d
"""
import os
import shutil
import cv2
import numpy as np
import pandas as pd

from tactile_data.tactile_servo_control import BASE_DATA_PATH
from tactile_data.utils_data import save_json_obj, load_json_obj, make_dir
from tactile_image_processing.image_transforms import process_image
from tactile_servo_control.utils.setup_parse_args import setup_parse_args


def split(path, dir_in_str, dirs_out, frac=0.8):

    # load target df
    targets_df = pd.read_csv(os.path.join(path, dir_in_str, 'targets.csv'))

    # indices to split data
    np.random.seed(1) # make predictable needs to be different from collect
    inds_true = np.random.choice([True, False], size=len(targets_df), p=[frac, 1-frac])
    inds = [inds_true, ~inds_true]

    # iterate over split
    for dir_out_str, ind in zip(dirs_out, inds):
        dir_in = os.path.join(path, dir_in_str)
        dir_out = os.path.join(path, dir_out_str)

        # if new directory, copy collect, env and sensor parameters
        if dir_out != dir_in:
            make_dir(dir_out, check=False)
            shutil.copy(os.path.join(dir_in, 'collect_params.json'), dir_out)
            shutil.copy(os.path.join(dir_in, 'env_params.json'), dir_out)
            shutil.copy(os.path.join(dir_in, 'sensor_params.json'), dir_out)  
        
            # create dataframe pointing to original images (to avoid copying)
            targets_df.loc[ind, 'sensor_image'] = \
                rf'../../{dir_in_str}/images/' + targets_df[ind].sensor_image.map(str)
            targets_df[ind].to_csv(os.path.join(dir_out, 'targets.csv'), index=False)


def process(path, dirs, process_params={}):

    # iterate over dirs
    for dir in dirs:

        # paths
        image_dir = os.path.join(path, dir, 'images')
        proc_image_dir = os.path.join(path, dir, 'processed_images')
        os.makedirs(proc_image_dir, exist_ok=True)
        
        # process images
        targets_df = pd.read_csv(os.path.join(path, dir, 'targets.csv'))
        for sensor_image in targets_df.sensor_image:
            print(f'processed {dir}: {sensor_image}')

            image = cv2.imread(os.path.join(image_dir, sensor_image))
            proc_image = process_image(image, **process_params)
            image_path, proc_sensor_image = os.path.split(sensor_image)
            cv2.imwrite(os.path.join(proc_image_dir, proc_sensor_image), proc_image)

        # if targets have paths remove them
        if image_path:
            targets_df.loc[:, 'sensor_image'] = \
                targets_df.sensor_image.str.split('/', expand=True).iloc[:,-1]
            targets_df.to_csv(os.path.join(path, dir, 'targets.csv'), index=False)

        # save merged sensor_params and process_params
        sensor_params = load_json_obj(os.path.join(path, dir, 'sensor_params'))
        sensor_proc_params = {**sensor_params, **process_params}

        if 'bbox' in sensor_params and 'bbox' in sensor_proc_params:
            b, pb = sensor_params['bbox'], sensor_proc_params['bbox']
            sensor_proc_params['bbox'] = [b[0]+pb[0], b[1]+pb[1], b[0]+pb[2], b[1]+pb[3]]
        
        save_json_obj(sensor_proc_params, os.path.join(path, dir, 'sensor_process_params'))


def main(
    robot='cr', 
    sensor='tactip_331',
    tasks=['edge_5d']
):
    robot_str, sensor_str, tasks, _, _, _ = setup_parse_args(robot, sensor, tasks)

    dir_in = "data"
    dirs_out = ["train", "val"]
    frac = 0.8

    process_params = {
        'thresh': True,
        'dims': (128, 128),
        "circle_mask_radius": 220,
        "bbox": (10, 10, 430, 430) # sim (12, 12, 240, 240) # midi (10, 10, 430, 430) # mini (10, 10, 310, 310)
    }

    for task in tasks:
        path = os.path.join(BASE_DATA_PATH, robot_str+'_'+sensor_str, task)
        split(path, dir_in, dirs_out, frac)
        process(path, dirs_out, process_params)


if __name__ == "__main__":
    main()
