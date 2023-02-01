import os

from tactile_gym.utils.general_utils import save_json_obj

from tactile_gym_servo_control.collect_data.utils_collect_data import make_target_df_rand
from tactile_gym_servo_control.collect_data.utils_collect_data import create_data_dir


def setup_sensor(
    collect_dir
):
    sensor_params = {
        "name": "tactip",
        "type": "standard",
        "core": "no_core",
        "dynamics": {},
        "image_size": [128, 128],
        "turn_off_border": False,
    }

    save_json_obj(sensor_params, os.path.join(collect_dir, 'sensor_params'))

    return sensor_params


def setup_surface_3d_collect_data(
    collect_dir,
    num_samples=10,
    shuffle_data=False,
):
    env_params = {
        'stim_name': 'square',
        'stim_pose': [600, 0, 12.5,    0, 0,  0],
        'workframe': [600, 0, 52.5, -180, 0, 90]
    }

    pose_params = {
        'pose_llims': [0, 0, 0.5, -25, -25, 0],
        'pose_ulims': [0, 0, 5.5,  25,  25, 0],
        'obj_poses': [[0, 0, 0, 0, 0, 0]],
    }

    target_df = make_target_df_rand(
        num_samples, shuffle_data, **pose_params
    )

    image_dir = create_data_dir(collect_dir, target_df)

    sensor_params = setup_sensor(collect_dir)

    save_json_obj(pose_params, os.path.join(collect_dir, 'pose_params'))
    save_json_obj(env_params, os.path.join(collect_dir, 'env_params'))

    return target_df, image_dir, env_params, sensor_params


def setup_edge_2d_collect_data(
    collect_dir,
    num_samples=10,
    shuffle_data=False,
):
    env_params = {
        'stim_name': 'square',
        'stim_pose': [600, 0, 12.5,    0, 0, 0],
        'workframe': [650, 0, 52.5, -180, 0, 0]
    }

    pose_params = {
        'pose_llims': [-5, 0, 2.5, 0, 0, -180],
        'pose_ulims': [5, 0, 3.5, 0, 0,  180],
        'obj_poses': [[0, 0, 0, 0, 0, 0]]
    }

    target_df = make_target_df_rand(
        num_samples, shuffle_data, **pose_params
    )

    image_dir = create_data_dir(collect_dir, target_df)

    sensor_params = setup_sensor(collect_dir)

    save_json_obj(pose_params, os.path.join(collect_dir, 'pose_params'))
    save_json_obj(env_params, os.path.join(collect_dir, 'env_params'))

    return target_df, image_dir, env_params, sensor_params


def setup_edge_3d_collect_data(
    collect_dir,
    num_samples=10,
    shuffle_data=False,
):
    env_params = {
        'stim_name': 'square',
        'stim_pose': [600, 0, 12.5, 0, 0, 0],
        'workframe': [650, 0, 52.5, -180, 0, 90]
    }

    pose_params = {
        'pose_llims': [0, -3, 2,  -20, -20, -180],
        'pose_ulims': [0,  3, 5.5, 20,  20,  180],
        'obj_poses': [[0, 0, 0, 0, 0, 0]]
    }

    target_df = make_target_df_rand(
        num_samples, shuffle_data, **pose_params
    )

    image_dir = create_data_dir(collect_dir, target_df)

    sensor_params = setup_sensor(collect_dir)

    save_json_obj(pose_params, os.path.join(collect_dir, 'pose_params'))
    save_json_obj(env_params, os.path.join(collect_dir, 'env_params'))

    return target_df, image_dir, env_params, sensor_params


def setup_edge_5d_collect_data(
    collect_dir,
    num_samples=10,
    shuffle_data=False,
):
    env_params = {
        'stim_name': 'square',
        'stim_pose': [600, 0, 12.5, 0, 0, 0],
        'workframe': [650, 0, 52.5, -180, 0, 90]
    }

    pose_params = {
        'pose_llims': [0, -4,   2, -15, -15, -180],
        'pose_ulims': [0,  4, 5.5,  15,  15,  180],
        'obj_poses': [[0, 0, 0, 0, 0, 0]]
    }

    target_df = make_target_df_rand(
        num_samples, shuffle_data, **pose_params
    )

    image_dir = create_data_dir(collect_dir, target_df)

    sensor_params = setup_sensor(collect_dir)

    save_json_obj(pose_params, os.path.join(collect_dir, 'pose_params'))
    save_json_obj(env_params, os.path.join(collect_dir, 'env_params'))

    return target_df, image_dir, env_params, sensor_params


setup_collect_data = {
    "surface_3d": setup_surface_3d_collect_data,
    "edge_2d": setup_edge_2d_collect_data,
    "edge_3d": setup_edge_3d_collect_data,
    "edge_5d": setup_edge_5d_collect_data
}


if __name__ == '__main__':
    pass
