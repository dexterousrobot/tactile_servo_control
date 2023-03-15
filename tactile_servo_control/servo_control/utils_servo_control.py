# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch.autograd import Variable

import matplotlib
matplotlib.use("TkAgg") # change backend to stop plt stealing focus
import matplotlib.pylab as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from cri.transforms import quat2euler, euler2quat, inv_transform
from tactile_servo_control.learning.utils_learning import POSE_LABEL_NAMES
from tactile_image_processing.image_transforms import process_image


class PoseModel:
    def __init__(self,
        model, 
        image_processing_params, 
        label_encoder,
        device='cpu'
    ):
        self.model = model
        self.image_processing_params = image_processing_params
        self.pose_encoder = label_encoder 
        self.label_names = label_encoder.target_label_names
        self.device = device

    def predict(self, tactile_image):

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
        outputs = self.model(model_input)

        # decode the prediction
        predictions_dict = self.pose_encoder.decode_label(outputs)

        print("\n Predictions: ", end="")
        predictions_arr = np.zeros(6)
        for label_name in self.label_names:
            predicted_val = predictions_dict[label_name].detach().cpu().numpy() 
            predictions_arr[POSE_LABEL_NAMES.index(label_name)] = predicted_val
            
            with np.printoptions(precision=2, suppress=True):
                print(label_name, predicted_val, end=" ")

        return predictions_arr


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK; can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


class PlotContour3D:
    def __init__(self, 
                workframe=[0, 0, 0, 0, 0, 180],
                stim_name=None,
                position=[0, 400],
                r=[-1,1]/np.sqrt(2),
                inv=1
                ):

        if stim_name=='saddle':
            limits = [[-70,70], [-10,130], [-30,30]]
        else:
            limits = [[-110,10], [-60,60], [-30,30]]

        self._fig = plt.figure('Contour 3d', figsize=(5, 5))
        self._fig.clear()
        self._fig.subplots_adjust(left=-0.1, right=1.1, bottom=-0.05, top=1.05)
        self._ax = self._fig.add_subplot(111, projection='3d')
        # self._ax.view_init(90,-90, 0)#(elev=30, azim=45, roll=15)
        self._ax.view_init(30, 45, 15, 'z')
        self._ax.azim = workframe[5]
        self._ax.plot(limits[0], limits[1], limits[2], ':w')  

        self.r = r 
        self.inv = inv 
        self.v = [0, 0, 0, 0, 0, 0]

        move_figure(self._fig, *position)
        self._fig.show()

    def update(self, v):
        self.v = np.vstack([self.v, v])

        v_q = euler2quat([0, 0, 0, *self.v[-1,3:]], axes='rxyz')
        d_q = euler2quat([*self.r[::-1], 0, 0, 0, 0], axes='rxyz')
        w_q = euler2quat([0, 0, -1, 0, 0, 0], axes='rxyz')
        d = 5*quat2euler(inv_transform(d_q, v_q), axes='rxyz')
        w = 5*quat2euler(inv_transform(w_q, v_q), axes='rxyz')

        self._ax.plot(
            self.inv*self.v[-2:,0], -self.v[-2:,1], -self.v[-2:,2], 
            '-r') 
        self._ax.plot(
            self.inv*self.v[-2:,0]+[d[0],-d[0]], -self.v[-2:,1]-[d[1],-d[1]], -self.v[-2:,2]-[d[2],-d[2]], 
            '-b', linewidth=0.5) 
        self._ax.plot(
            self.inv*self.v[-2:,0]+[w[0],0], -self.v[-2:,1]-[w[1],0], -self.v[-2:,2]-[w[2],0], 
            '-g', linewidth=0.5) 

        self._fig.canvas.flush_events()   # update the plot
        plt.pause(0.0001)

    def save(self, outfile=None):
        if outfile:
            self._fig.savefig(outfile, bbox_inches="tight") 


if __name__ == '__main__':
    pass
