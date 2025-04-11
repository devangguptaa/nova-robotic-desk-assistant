import cv2
from uvc_camera import UVCCamera
from camera_callibration import CameraCalibrate
import stag
import numpy as np

import json
import time
from marker_utils import *
from ultralytics import YOLO
from scipy.linalg import svd
from pymycobot import *
from pump import PumpControl


class MoveArm():
    def __init__(self,mc):
        self.origin = [0,0,0,0,0,0]
        self.me = [-63.36, -74.44, 30, 3.07, -7.2, 48.51]
        self.observe = [19.07, -0.17, -10.45, -57.83, 1.58, 28.3]
        self.mc = mc 

    def wait(self):
        time.sleep(0.5)
        while(self.mc.is_moving() == 1):
            time.sleep(0.2)

    def move_to_position(self, position):
        if (position == "me" or position == "to_me"):
            self.mc.send_angles(self.me, 30)
            self.wait()
        elif (position == "observe"): 
            self.mc.send_angles(self.observe, 30)
            self.wait()
        elif (position == "origin"):
            print("move to origin")
            self.mc.send_angles(self.origin, 30)
            self.wait()

