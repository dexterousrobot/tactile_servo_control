import os
import numpy as np
import pandas as pd


def setup_target_df(
    task_params,
    num_poses=100, 
    save_dir=None,
):

    # generate random poses 
    np.random.seed(0) # make predictable
    poses = np.random.uniform(
        task_params['pose_llims'], task_params['pose_ulims'], size=(num_poses, 6)
    )
    shears = np.random.uniform(
        task_params['shear_llims'], task_params['shear_ulims'], size=(num_poses, 6)
    )
    
    # sort parameters by label
    if task_params.get('sort', False):
        ind = task_params['pose_label_names'].index(task_params['sort'])
        poses = poses[poses[:, ind].argsort()]

    # generate and save target data
    target_df = pd.DataFrame(
        columns=[
            "sensor_image",
            *task_params['pose_label_names'],
            *task_params['shear_label_names']
        ]
    )

    # populate dataframe
    for i in range(num_poses):
        image_name = f"image_{i+1}.png"
        pose = poses[i,:]
        shear = shears[i,:]
        target_df.loc[i] = np.hstack((image_name, pose, shear))

    # shuffle        
    if save_dir:
        target_file = os.path.join(save_dir, "targets.csv")
        target_df.to_csv(target_file, index=False)

    return target_df
