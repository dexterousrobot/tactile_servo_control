import os
import numpy as np
import torch
from torch.autograd import Variable

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tactile_gym_servo_control.learning.utils_learning import decode_pose
from tactile_gym_servo_control.learning.utils_learning import POSE_LABEL_NAMES
from tactile_gym_servo_control.utils.image_transforms import process_image

LEFT, RIGHT, FORE, BACK, SHIFT, CTRL, QUIT \
    = 65295, 65296, 65297, 65298, 65306, 65307, ord('Q')


def keyboard(embodiment,  
    delta_init = [0.0, 0, 0, 0, 0, 0]
):
    delta = np.array([0, 0, 0, 0, 0, 0]) + delta_init # stop keeping state

    keys = embodiment._pb.getKeyboardEvents()
    if embodiment._pb.KEY_WAS_TRIGGERED:
        if CTRL in keys:
            if FORE in keys:  delta -= [0, 0, 0, 0, 1, 0]
            if BACK in keys:  delta += [0, 0, 0, 0, 1, 0]
            if RIGHT in keys: delta -= [0, 0, 0, 1, 0, 0]
            if LEFT in keys:  delta += [0, 0, 0, 1, 0, 0]
        elif SHIFT in keys:
            if FORE in keys:  delta -= [0, 0, 1, 0, 0, 0]
            if BACK in keys:  delta += [0, 0, 1, 0, 0, 0]
            if RIGHT in keys: delta -= [0, 0, 0, 0, 0, 2.5]
            if LEFT in keys:  delta += [0, 0, 0, 0, 0, 2.5]
        else:
            if FORE in keys:  delta -= [1, 0, 0, 0, 0, 0]
            if BACK in keys:  delta += [1, 0, 0, 0, 0, 0]
            if RIGHT in keys: delta -= [0, 1, 0, 0, 0, 0]
            if LEFT in keys:  delta += [0, 1, 0, 0, 0, 0]
        if QUIT in keys:  delta = None

    return np.array(delta)


class Slider:
    def __init__(self,
        embodiment, init_ref_pose,
        ref_llims=[-5, -5, 0, -15, -15, -180],
        ref_ulims=[ 5,  5, 5,  15,  15,  180]
    ):    
        self.embodiment = embodiment
        self.ref_pose_ids = []
        for label_name in POSE_LABEL_NAMES:
            i = POSE_LABEL_NAMES.index(label_name)
            self.ref_pose_ids.append(
                embodiment._pb.addUserDebugParameter(
                    label_name, ref_llims[i], ref_ulims[i], init_ref_pose[i]
                )
            )

    def slide(self, ref_pose):
        for j in range(len(ref_pose)):
            ref_pose[j] = self.embodiment._pb.readUserDebugParameter(
                self.ref_pose_ids[j]
            ) 

        return ref_pose


class Model:
    def __init__(self,
        model, image_processing_params, pose_params, 
        label_names, 
        device='cpu'
    ):
        self.model = model
        self.image_processing_params = image_processing_params
        self.label_names = label_names 
        self.pose_limits = [pose_params['pose_llims'], pose_params['pose_ulims']]
        self.device = device

    def predict(self, 
        tactile_image
    ):
        processed_image = process_image(
            tactile_image,
            gray=False,
            **self.image_processing_params
        )

        # channel first for pytorch
        processed_image = np.rollaxis(processed_image, 2, 0)

        # add batch dim
        processed_image = processed_image[np.newaxis, ...]

        # perform inference with the trained model
        model_input = Variable(torch.from_numpy(processed_image)).float().to(self.device)
        raw_predictions = self.model(model_input)

        # decode the prediction
        predictions_dict = decode_pose(raw_predictions, self.label_names, self.pose_limits)

        print("\nPredictions: ", end="")
        predictions_arr = np.zeros(6)
        for label_name in self.label_names:
            predicted_val = predictions_dict[label_name].detach().cpu().numpy() 
            predictions_arr[POSE_LABEL_NAMES.index(label_name)] = predicted_val
            print(label_name, predicted_val, end=" ")

        return predictions_arr
