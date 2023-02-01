import os

from tactile_gym.utils.general_utils import save_json_obj

from tactile_gym_servo_control.collect_data.utils_collect_data import make_target_df_rand
from tactile_gym_servo_control.collect_data.utils_collect_data import create_data_dir

data_path = os.path.join(os.path.dirname(__file__), '../../example_data/real')


def setup_sensor(
    collect_dir
):
    sensor_params = {
        'source': 0,
        'exposure': -7,
        'gray': True,
        'bbox': None,
        'thresh': None,
        'circle_mask_radius': None
    }

    save_json_obj(sensor_params, os.path.join(collect_dir, 'sensor_params'))

    return sensor_params


def setup_edge_2d(
    collect_dir,
    num_samples=10,
    shuffle_data=True,
):
    env_params = {
        'workframe': [285, 0, -93, 0, 0, 180],
        'linear_speed': 10,
        'angular_speed': 10,
        'tcp_pose': [0, 0, 0, 0, 0, 0]
    }

    pose_params = {
        'pose_llims': [-5, 0, -1, 0, 0, -180],
        'pose_ulims': [5, 0,  1, 0, 0,  180],
        'move_llims': [-5, -5,  0, 0, 0,  -5],
        'move_ulims': [5,  5,  0, 0, 0,   5],
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


def setup_edge_3d(
    collect_dir,
    num_samples=10,
    shuffle_data=True,
):
    env_params = {
        'workframe': [285, 0, -93, 0, 0, 180],
        'linear_speed': 10,
        'angular_speed': 10,
        'tcp_pose': [0, 0, 0, 0, 0, 0]
    }

    pose_params = {
        'pose_llims': [-5, 0, 0.5, 0, 0, -90],
        'pose_ulims': [5, 0,   5, 0, 0,  90],
        'move_llims': [-5, -5, 0, 0, 0, -5],
        'move_ulims': [5,  5, 0, 0, 0,  5],
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
    "edge_2d": setup_edge_2d,
    "edge_3d": setup_edge_3d
}


if __name__ == '__main__':
    pass
