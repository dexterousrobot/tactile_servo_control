import numpy as np
import torch
from torch.autograd import Variable

from tactile_image_processing.image_transforms import process_image


class LabelledModel:
    def __init__(self,
                 model,
                 image_processing_params,
                 label_encoder,
                 device='cuda'
                 ):
        self.model = model
        self.image_processing_params = image_processing_params
        self.label_encoder = label_encoder
        self.label_names = label_encoder.label_names
        self.target_label_names = label_encoder.target_label_names
        self.device = device

    def predict(self, tactile_image):

        processed_image = process_image(
            tactile_image,
            gray=False,
            **self.image_processing_params
        )

        # channel first for pytorch; add batch dim
        processed_image = np.rollaxis(processed_image, 2, 0)
        processed_image = processed_image[np.newaxis, ...]

        # perform inference with the trained model
        model_input = Variable(torch.from_numpy(processed_image)).float().to(self.device)
        outputs = self.model(model_input)

        # decode the prediction
        predictions_dict = self.label_encoder.decode_label(outputs)

        # pack into array and report
        print("\nPredictions: ", end="")
        predictions_arr = np.zeros(len(self.label_names))
        for label_name in self.target_label_names:
            predicted_val = predictions_dict[label_name].detach().cpu().numpy()
            predictions_arr[self.label_names.index(label_name)] = predicted_val
            with np.printoptions(precision=2, suppress=True):
                print(label_name, predicted_val, end=" ")

        return predictions_arr


if __name__ == '__main__':
    pass
