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


def process(
    robot='sim', 
    sensor='tactip',
    tasks=['edge_5d']
):
    dir_in_str = "data"
    dirs_out_str = ["train", "val"]
    split = 0.8

    # optional image processing
    sensor_params = {
        # 'thresh': True,
        # 'dims': (128,128),
        # "circle_mask_radius": 220,
        # "bbox": (10, 10, 430, 430) # "bbox": (10, 10, 310, 310)
    }

    robot_str, sensor_str, tasks, _, _, _ = setup_parse_args(robot, sensor, tasks)

    for task_str in tasks:

        # load target df
        path = os.path.join(BASE_DATA_PATH, robot_str+'_'+sensor_str, task_str)
        targets_df = pd.read_csv(os.path.join(path, dir_in_str, 'targets.csv'))

        # indices to split data
        np.random.seed(0) # make predictable
        inds_true = np.random.choice([True, False], size=len(targets_df), p=[split, 1-split])
        inds = [inds_true, ~inds_true]

        # iterate over split
        for dir_out_str, i in zip(dirs_out_str, inds):

            dir_in = os.path.join(path, dir_in_str)
            dir_out = os.path.join(path, dir_out_str)

            # copy task and env parameters
            make_dir(dir_out, check=False)
            shutil.copy(os.path.join(dir_in, 'task_params.json'), dir_out)
            shutil.copy(os.path.join(dir_in, 'env_params.json'), dir_out)
            
            # process image data if new sensor_params supplied
            if sensor_params: 
                
                # create new image directory
                image_dir_in = os.path.join(dir_in, 'images')
                image_dir_out = os.path.join(dir_out, 'processed_images')
                os.makedirs(image_dir_out, exist_ok=True)
                
                # merge new and old sensor_params
                sensor_params_in = load_json_obj(os.path.join(dir_in, 'sensor_params'))
                sensor_params_out = {**sensor_params_in, **sensor_params}
                
                if 'bbox' in sensor_params_in and 'bbox' in sensor_params:
                    b, b_in = sensor_params['bbox'], sensor_params_in['bbox']
                    sensor_params_out['bbox'] = [b_in[0]+b[0], b_in[1]+b[1], b_in[0]+b[2], b_in[1]+b[3]]

                save_json_obj(sensor_params_out, os.path.join(dir_out, 'sensor_params'))

                # populate with images
                for sensor_image in targets_df[i].sensor_image:
                    print(f'processed {dir_out_str}: {sensor_image}')
                    image = cv2.imread(os.path.join(image_dir_in, sensor_image))
                    processed_image = process_image(image, **sensor_params)
                    cv2.imwrite(os.path.join(image_dir_out, sensor_image), processed_image)
            
            # or just point to original data if no new sensor_params
            else:
                shutil.copy(os.path.join(dir_in, 'sensor_params.json'), dir_out)  
                targets_df.loc[i,'sensor_image'] = \
                    rf'../../{dir_in_str}/images/' + targets_df[i].sensor_image.map(str)
            
            # save targets
            targets_df[i].to_csv(os.path.join(dir_out, 'targets.csv'), index=False)


if __name__ == "__main__":
    process()
