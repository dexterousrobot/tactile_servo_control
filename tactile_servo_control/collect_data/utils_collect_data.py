import os
import numpy as np
import pandas as pd


def setup_target_df(
    task_params,
    num_poses=100, 
    save_dir=None,
):

    pose_lims = [task_params['pose_llims'], task_params['pose_ulims']]
    shear_lims = [task_params['shear_llims'], task_params['shear_ulims']]
    sample_disk = task_params.get('sample_disk', False)

    # generate random poses 
    np.random.seed(0) # make predictable
    poses = sample_poses(*pose_lims, num_poses, sample_disk)
    shears = sample_poses(*shear_lims, num_poses, sample_disk)

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

    # save to file        
    if save_dir:
        target_file = os.path.join(save_dir, "targets.csv")
        target_df.to_csv(target_file, index=False)

    return target_df


def random_spherical(num_samples, phi_max):   # phi_max degrees
    # Return uniform random sample over a spherical cap bounded by polar angle

    phi_max = np.radians(phi_max)                                 # maximum value of polar angle
    theta = 2*np.pi * np.random.rand(num_samples)                 # azimuthal angle samples
    kappa = 0.5 * (1 - np.cos(phi_max))                           # value of cumulative dist function at phi_max
    phi = np.arccos(1 - 2 * kappa * np.random.rand(num_samples))  # polar angle samples
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Compute Rx, Ry component samples for extrinsic-xyz Euler parameterization
    Rx = -np.arcsin(y)        # Rotation around x needed to move (0, 0, 1) to (*, y, *)
    Ry = -np.arctan2(x, z)    # Rotation around y needed to move (*, y, *) to (x, y, z) (r = 1)

    return np.degrees(Rx), np.degrees(Ry)   # degrees


def random_disk(num_samples, r_max):
    # Return uniform random sample over a 2D circular disk of radius r_max

    theta = 2*np.pi * np.random.rand(num_samples)
    r = r_max * np.sqrt(np.random.rand(num_samples))

    x, y = r * (np.cos(theta), np.sin(theta))
    theta = np.degrees(theta)

    return x, y


def random_linear(num_samples, x_max):
    # Return uniform random sample over a 1D segment [-x_max, x_max]

    x = -x_max + 2 * x_max * np.random.rand(num_samples)

    return x


def sample_poses(llims, ulims, num_samples, sample_disk):

    poses_mid = ( np.array(ulims) + llims ) / 2
    poses_max = ulims - poses_mid

    # default linear sampling on all components
    samples = [random_linear(num_samples, x_max) for x_max in poses_max]

    # resample components if circular sampling
    if sample_disk:
        inds_pos = [i for i,v in enumerate(poses_max[:3]) if v>0]     # any x, y, z
        inds_rot = [3+i for i,v in enumerate(poses_max[3:5]) if v>0]  # only Rx, Ry

        if len(inds_pos) >= 2:
            r_max = max(poses_max[inds_pos])  
            samples_pos = random_disk(num_samples, r_max)

            scales = poses_max[inds_pos] / r_max # for limits not equal
            samples_pos *= scales[np.newaxis,:2].T

            samples[inds_pos[0]], samples[inds_pos[1]] = samples_pos

        if len(inds_rot) == 2:
            phi_max = max(poses_max[inds_rot]) 
            samples_rot = random_spherical(num_samples, phi_max)

            scales = poses_max[inds_rot[:2]] / phi_max # for limits not equal
            samples_rot *= scales[np.newaxis,:].T

            samples[inds_rot[0]], samples[inds_rot[1]] = samples_rot

    poses = np.array(samples).T
    poses += poses_mid

    return poses
    