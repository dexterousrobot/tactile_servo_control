import numpy as np

from tactile_gym_servo_control.utils_robot_real.setup_pybullet_env import setup_pybullet_env
from vsp.video_stream import CvImageOutputFileSeq, CvVideoDisplay, CvPreprocVideoCamera   
from vsp.processor import CameraStreamProcessor, AsyncProcessor

from cri.robot import SyncRobot
from cri.controller import Mg400Controller as Controller
# from cri.controller import DummyController as Controller

def make_sensor(
    size=[256, 256], 
    bbox=None, 
    exposure=-7, 
    source=0, 
    thresh=True,
    **kwargs
):  
    camera = CvPreprocVideoCamera(
        size, crop=bbox, threshold=[61, -5], exposure=exposure, source=source
    )
    
    for _ in range(5): camera.read() # Hack - camera transient   
    
    return AsyncProcessor(CameraStreamProcessor(
            camera=camera,
            display=CvVideoDisplay(name='sensor'),
            writer=CvImageOutputFileSeq())
    )


def setup_embodiment_env(
    sensor_params={},
    workframe=[288, 0, -100, 0, 0, -90],
    linear_speed=10, 
    angular_speed=10,
    tcp_pose=[0, 0, 0, 0, 0, 0],
    hover=[0, 0, 10, 0, 0, 0], # positive for dobot
    show_gui=True
):

    # setup the tactile sensor
    sensor = make_sensor(**sensor_params)

    def sensor_process(outfile=None):
        img = sensor.process(
            num_frames=1, start_frame=1, outfile=outfile
        )
        return img[0,:,:,0]

    # setup the robot
    embodiment = SyncRobot(Controller())
    embodiment.sensor_process = sensor_process
    embodiment.slider = setup_pybullet_env(show_gui)
    embodiment.sim = False

    embodiment.coord_frame = workframe
    embodiment.linear_speed = linear_speed
    embodiment.angular_speed = angular_speed
    embodiment.tcp = tcp_pose

    embodiment.hover = np.array(hover)
    embodiment.show_gui = show_gui
    embodiment.workframe = workframe

    return embodiment


if __name__ == '__main__':
    pass
