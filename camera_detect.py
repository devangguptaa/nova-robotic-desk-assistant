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

# import serial.tools.list_ports

# def find_arduino_port():
#     print("here")
#     ports = serial.tools.list_ports.comports()
#     for port in ports:
#         # Check if the port description or device name contains typical Arduino indicators
#         if "Arduino" in port.description or "usbmodem" in port.device:
#             return port.device
#     return None

# arduino_port = find_arduino_port()

# if arduino_port:
#     print("Arduino found at:", arduino_port)
#     mc = MyCobot280(arduino_port, 115200)  
# else:
#     print("Arduino not found!")

# # mc = MyCobot280("/dev/tty.usbmodem1101",115200) 
# type = mc.get_system_version()
offset_j5 = -90
# if type > 2:
#     offset_j5 = -90
#     print("280")
#     print(mc.get_movement_type())
# else:
#     print("320")

            
np.set_printoptions(suppress=True, formatter={'float_kind': '{:.2f}'.format})





class camera_detect:
    #Camera parameter initialize
    def __init__(self, camera_id, marker_size, mtx, dist,mc):
        self.camera_id = camera_id
        self.mtx = mtx
        self.dist = dist
        self.marker_size = marker_size
        self.camera = UVCCamera(self.camera_id, self.mtx, self.dist)
        self.yolo_model = YOLO("yolov8s.pt")
        self.calibrate = CameraCalibrate(self.camera_id,marker_size,mtx,dist)
        # self.calibrate.camera_open()
        self.camera.capture()
        self.pump = PumpControl(mc)
        self.origin_mycbot_horizontal = [42.36, -35.85, -52.91, 88.59, 90+offset_j5, 0.0]
        self.origin_mycbot_level = [-90, 5, -104, 14, 90 + offset_j5, 0]
        self.IDENTIFY_LEN = 150 #to keep identify length
        self.home_left_angles = [-63.36, -74.44, 20, 3.07, -7.2, 48.51]
        self.out_file = 1
        self.mc = mc
        # Initialize EyesInHand_matrix to None or load from a document if available
        self.EyesInHand_matrix = None
        self.load_matrix()

    def load_matrix(self, filename="EyesInHand_matrix.json"):
        # Load the EyesInHand_matrix from a JSON file, if it exists
        try:
            with open(filename, 'r') as f:
                self.EyesInHand_matrix = np.array(json.load(f))
        except FileNotFoundError:
            print("Matrix file not found. EyesInHand_matrix will be initialized later.")
    def wait(self):
        time.sleep(0.5)
        while(self.mc.is_moving() == 1):
            time.sleep(0.2)
    
    def drop(self):
        self.pump.pump_off()

    def safe_get_coords(self,ml, max_retries=10, delay=0.5):
        coords = ml.get_coords()
        retry_count = 0

        while isinstance(coords, int) and coords == -1 and retry_count < max_retries:
            print(f"Warning: MyCobot returned -1 (communication error). Retrying... ({retry_count+1}/{max_retries})")
            time.sleep(delay)  # Wait before retrying
            coords = ml.get_coords()
            retry_count += 1

        if isinstance(coords, int) and coords == -1:
            print("Error: MyCobot communication failed after multiple attempts. Returning None.")
            return None

        return coords
    
    def coord_limit(self, coords):
        min_coord = [-350, -350, 300]
        max_coord = [350, 350, 500]
        for i in range(3):
            if(coords[i] < min_coord[i]):
                coords[i] = min_coord[i]

            if(coords[i] > max_coord[i]):
                coords[i] = max_coord[i]
        return coords

    def vision_trace(self, mode, ml):
        # input("enter any key start vision trace")
        sp = 40 

        if mode == 0: 
            ml.send_angles(self.origin_mycbot_horizontal, sp) 
            self.wait(ml) 
            input("enter any key to start trace")
            
            target_coords,_ = self.stag_robot_identify(ml)
            print(target_coords)

            time.sleep(1)
            ml.send_coords(target_coords, 30) 
            self.wait(ml) 

    def detect_markers_in_yolo_bbox(self,objects):
        # Get the current frame from the camera
        frame = self.camera.color_frame()

        # Run YOLO on the frame to get object detections

        """Model classes: {0: 'person'ß, 24: 'backpack', 25: 'umbrella', 26: 'handbag', 39: 'bottle', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 58: 'potted plant', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors'}"""

        results = self.yolo_model(frame,conf=0.5,classes=objects)  # Only detect class 0 (marker)
        
        # Depending on the YOLO version, you might need to adjust how you access the results.
        markers_list = []
        for result in results:
            # Check if there are any detections
            if len(result.boxes.xyxy) == 0:
                continue  # Skip this result if no detections
            
            # If there are detections, iterate through each bounding box
            for box in result.boxes.xyxy:
                bbox = box.cpu().numpy().astype(int)
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Crop the frame to the detected bounding box
                cropped_frame = frame[y1:y2, x1:x2]
                self.out_file += 1
                # Run stag marker detection on the cropped region
                corners, ids, _ = stag.detectMarkers(cropped_frame, 11)
                if corners is not None and len(corners) > 0:
                    success = cv2.imwrite(f"{self.out_file}.jpg", cropped_frame)

                    # Adjust the marker corner coordinates back to the original frame coordinates
                    adjusted_corners = []
                    for corner in corners:
                        adjusted_corner = corner + np.array([[x1, y1]])
                        adjusted_corners.append(adjusted_corner)
                    
                    markers_list.append((adjusted_corners, ids))
    
            return markers_list
    
    def vision_trace_loop_yolo(self, ml,objects,direction):
        ml.set_fresh_mode(1)
        time.sleep(1)
        # ml.send_angles(self.origin_mycbot_horizontal, 50)  # 移动到观测点
        time.sleep(2)
        # ml.send_angles(self.origin_mycbot_horizontal, 50)  # 移动到观测点
        # mc.send_angles([19.07, -0.17, -10.45, -57.83, 1.58, 28.3],20)

        # ml.send_angles([-97.73, -1.58, -0.96, 52.38, 166.99, 52.64],50)
        # ml.sync_send_angles([-97.73, -1.58, -0.96, 52.38, 165, 52.64], 50)
        ml.send_angles([19.07, -0.17, -10.45, -63.83, 1.58, 28.3],50)
        self.wait()
        origin = self.safe_get_coords(ml,10,0.5)
        while origin is None:
            origin = self.safe_get_coords(ml,10,0.5)
        # time.sleep(1)
        flag = False 
        ctr = 0 
        while (not flag) and ctr<=50:
            ctr +=1
            self.camera.update_frame()
            frame = self.camera.color_frame()
            
            # Use YOLO to restrict detection to relevant ROIs
            markers_in_boxes = self.detect_markers_in_yolo_bbox(objects)

            if markers_in_boxes == None:
                print("No markers detected in YOLO bounding boxes.")
                continue
            
            # Process each set of detected markers
            for marker_corners, ids in markers_in_boxes:
                # Here you can calculate the marker position based on adjusted corners
                marker_pos_pack = self.calibrate.calc_markers_base_position(marker_corners, ids)
                
                # Get current robot coords and transform the marker coordinates
                target_coords = self.safe_get_coords(ml,10,0.5)
                while target_coords is None:
                    target_coords = self.safe_get_coords(ml,10,0.5)
                cur_coords = np.array(target_coords.copy())
                cur_coords[-3:] *= (np.pi / 180)
                print(cur_coords, marker_pos_pack)
                fact_bcl = self.calibrate.Eyes_in_hand(cur_coords, marker_pos_pack, self.EyesInHand_matrix)
                for i in range(3):
                    target_coords[i] = fact_bcl[i]
                print("Adjusted robot coords:", target_coords)
                target_coords = self.coord_limit(target_coords)
                print(target_coords)
                for i in range(3):
                    target_coords[i+3] = origin[i+3]
                target_coords[2] = target_coords[2] - 300 if target_coords[2] > 0 else target_coords[2] + 300
                print("t2",target_coords)
                # You can then send these coordinates to the robot
                ml.send_coords(target_coords, 30)
                self.wait()
                self.pump.pump_on()
                time.sleep(1)
                if (direction=="pick"): 
                # ml.send_angles(self.origin_mycbot_horizontal, 30)  # 移动到观测点
                    ml.send_angles([19.07, -0.17, -10.45, -57.83, 1.58, 28.3],50)
                    self.wait()
                elif (direction == "to_me"): 
                    ml.send_angles([0,0,0,0,0,0],40)
                    self.wait()
                    ml.send_angles(self.home_left_angles,30)
                    self.wait()
                    self.pump.pump_off()
                    ml.send_angles([19.07, -0.17, -10.45, -57.83, 1.58, 28.3],50)
                    self.wait()
                elif (direction == "right"):
                    target_coords[2] = target_coords[2] + 50 if target_coords[2] > 0 else target_coords[2] - 50
                    target_coords[0] = target_coords[0] + 100
                    target_coords = self.coord_limit(target_coords)

                    ml.send_coords(target_coords, 30)
                    self.wait() 
                    self.pump.pump_off() 
                    ml.send_angles([19.07, -0.17, -10.45, -57.83, 1.58, 28.3],50)
                    self.wait()
                elif (direction == "left"):
                    target_coords[2] = target_coords[2] + 50 if target_coords[2] > 0 else target_coords[2] - 50
                    target_coords[0] = target_coords[0] - 25
                    target_coords = self.coord_limit(target_coords)

                    ml.send_coords(target_coords, 30)
                    self.wait() 
                    self.pump.pump_off()
                    ml.send_angles([19.07, -0.17, -10.45, -57.83, 1.58, 28.3],50)
                    self.wait()
                flag = True 

            # cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF != 255:
                break
   
    def vision_trace_loop(self, ml):
        self.mc.set_fresh_mode(1)
        time.sleep(1)
        ml.send_angles(self.origin_mycbot_horizontal, 50) 
        time.sleep(2)
        # ml.send_angles(self.origin_mycbot_horizontal, 50) 
        # mc.send_angles([19.07, -0.17, -10.45, -57.83, 1.58, 28.3],20)

        # ml.send_angles([-97.73, -1.58, -0.96, 52.38, 166.99, 52.64],50)
        # ml.sync_send_angles([-97.73, -1.58, -0.96, 52.38, 165, 52.64], 50)
        ml.send_angles([19.07, -0.17, -10.45, -57.83, 1.58, 28.3],50)
        self.wait()
        origin = self.safe_get_coords(ml,10,0.5)
        while origin is None:
            origin = self.safe_get_coords(ml,10,0.5)
        time.sleep(1)
        while 1:
            _ ,ids = self.calibrate.stag_identify()
            if ids[0] == 0:
                self.camera.update_frame()
                frame = self.camera.color_frame() 
                cv2.imshow("Enter any key to exit", frame)

                target_coords,_ = self.calibrate.stag_robot_identify(ml)
                self.coord_limit(target_coords)
                print(target_coords)
                for i in range(3):
                    target_coords[i+3] = origin[i+3]
                target_coords[2] = target_coords[2] - 200 if target_coords[2] > 0 else target_coords[2] + 200
                print("t2",target_coords)
                ml.send_coords(target_coords, 30) 
                time.sleep(0.5)
            elif ids[0] == 1:
                ml.send_angles(self.origin_mycbot_horizontal, 50) 

 

if __name__ == "__main__":
    # if mc.is_power_on()==0:
    #     mc.power_on()
    camera_params = np.load("camera_params.npz")  # 相机配置文件
    mtx, dist = camera_params["mtx"], camera_params["dist"]
    m = camera_detect(0, 43, mtx, dist,mc)
    # mc.set_vision_mode(1)
    # m.vision_trace_loop(mc)
    # m.vision_trace_loop_yolo(mc)

