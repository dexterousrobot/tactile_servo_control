"""
python test_model.py -r sim -m simple_cnn -t edge_2d
"""
import os
import numpy as np

from tactile_data.tactile_servo_control import BASE_DATA_PATH, BASE_MODEL_PATH
from tactile_data.utils_data import load_json_obj
from tactile_learning.supervised.models import create_model
from tactile_servo_control.utils.setup_parse_args import setup_parse_args
from tactile_servo_control.utils.setup_embodiment import setup_embodiment

from utils_learning import LabelEncoder, LabelledModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def test_model(
    robot,
    sensor,
    pose_model
):
    # start 50mm above workframe origin
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.move_joints((*robot.joint_angles[:-1], 0))

    # drop 10mm to contact object
    clearance = (0, 0, 10, 0, 0, 0)

    # labels to loop through
    label_names = pose_model.label_names
    target_label_names = pose_model.target_label_names

    # main loop
    for label_name in target_label_names:
        ind = label_names.index(label_name)
        llim = pose_model.label_encoder.llims_np[ind]
        ulim = pose_model.label_encoder.ulims_np[ind]

        # reset between each label
        robot.move_linear(np.array((0, 0, 3.5, 0, 0, 0)) - clearance)
        robot.move_joints((*robot.joint_angles[:-1], 0))

        # report 
        print(f'\n\nComponent {label_name}:', end='')

        for v in np.arange(llim, ulim+(ulim-llim)/10, (ulim-llim)/10):
            pose = np.array((0, 0, 3.5, 0, 0, 0))
            pose[ind] = v

            # report pose
            with np.printoptions(precision=2, suppress=True):        
                print(f'\n\nPose: ', end='')
                for i, label_name in enumerate(label_names):
                    print(f' {label_name} [{pose[i]}] ', end='')

            # move to above new pose (avoid changing pose in contact with object)
            # robot.move_linear(pose - clearance)

            # move to target positon 
            robot.move_linear(pose)

            # collect and process tactile image
            tactile_image = sensor.process()
            pose_model.predict(tactile_image)

            # move back to above target positon 
            # robot.move_linear(pose - clearance)

    # finish 50mm above workframe origin after zero last joint 
    robot.move_joints((*robot.joint_angles[:-1], 0))
    robot.move_linear(np.array((0, 0, 3.5, 0, 0, 0)) - clearance)
    
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.close()


if __name__ == "__main__":

    robot, sensor, tasks, models, _, device = setup_parse_args(
        robot='cr', 
        sensor='tactip_331',
        tasks=['edge_5d'],
        models=['simple_cnn'],
        device='cuda'
    )

    model_version = ''

    # test the trained networks
    for model_type, task in zip(models, tasks):

        # set data and model dir
        data_dir = os.path.join(BASE_DATA_PATH, robot+'_'+sensor, task, 'train_+yaw')
        model_dir = os.path.join(BASE_MODEL_PATH, robot+'_'+sensor, task, model_type + model_version)

        # load parameters
        env_params = load_json_obj(os.path.join(data_dir, 'env_params'))
        task_params = load_json_obj(os.path.join(model_dir, 'task_params'))
        model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
        preproc_params = load_json_obj(os.path.join(model_dir, 'preproc_params'))
        sensor_params = load_json_obj(os.path.join(model_dir, 'sensor_params'))

        # create the label encoder/decoder
        label_encoder = LabelEncoder(task_params, device)

        # setup embodiment, network and model
        robot, sensor = setup_embodiment(
            env_params, 
            sensor_params
        )

        # create the model
        model = create_model(
            in_dim=preproc_params['image_processing']['dims'],
            in_channels=1,
            out_dim=label_encoder.out_dim,
            model_params=model_params,
            saved_model_dir=model_dir,
            device=device
        )
        model.eval()

        pose_model = LabelledModel(
            model,
            preproc_params['image_processing'],
            label_encoder,
            device
        )

        test_model(
            robot,
            sensor,
            pose_model
        )
