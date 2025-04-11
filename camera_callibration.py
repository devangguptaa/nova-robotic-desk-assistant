import cv2
from uvc_camera import UVCCamera
import stag
import numpy as np
import json
import time
from marker_utils import *
from ultralytics import YOLO
from scipy.linalg import svd
from pymycobot import *
import serial.tools.list_ports

def find_arduino_port():
    print("here")
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Check if the port description or device name contains typical Arduino indicators
        if "Arduino" in port.description or "usbmodem" in port.device:
            return port.device
    return None

arduino_port = find_arduino_port()

if arduino_port:
    print("Arduino found at:", arduino_port)
    mc = MyCobot280(arduino_port, 115200) 
else:
    print("Arduino not found!")
# mc = MyCobot280("/dev/tty.usbmodem1101",115200) 
type = mc.get_system_version()
offset_j5 = 0
if type > 2:
    offset_j5 = -90
    print("280")
    print(mc.get_movement_type())
else:
    print("320")

            
np.set_printoptions(suppress=True, formatter={'float_kind': '{:.2f}'.format})



class CameraCalibrate:
    #Camera parameter initialize
    def __init__(self, camera_id, marker_size, mtx, dist):
        self.camera_id = camera_id
        self.mtx = mtx
        self.dist = dist
        self.marker_size = marker_size
        self.camera = UVCCamera(self.camera_id, self.mtx, self.dist)
        self.camera_open()
        self.origin_mycbot_horizontal = [42.36, -35.85, -52.91, 88.59, 90+offset_j5, 0.0]
        self.origin_mycbot_level = [-90, 5, -104, 14, 90 + offset_j5, 0]
        self.IDENTIFY_LEN = 300 #to keep identify length
   
        # Initialize EyesInHand_matrix to None or load from a document if available
        self.EyesInHand_matrix = None
        self.load_matrix()

    def save_matrix(self, filename="EyesInHand_matrix.json"):
        # Save the EyesInHand_matrix to a JSON file
        if self.EyesInHand_matrix is not None:
            with open(filename, 'w') as f:
                json.dump(self.EyesInHand_matrix.tolist(), f)
    
    def load_matrix(self, filename="EyesInHand_matrix.json"):
        # Load the EyesInHand_matrix from a JSON file, if it exists
        try:
            with open(filename, 'r') as f:
                self.EyesInHand_matrix = np.array(json.load(f))
        except FileNotFoundError:
            print("Matrix file not found. EyesInHand_matrix will be initialized later.")

    def wait(self):
        time.sleep(0.5)
        while(mc.is_moving() == 1):
            time.sleep(0.2)
    
    def camera_open(self):
        self.camera.capture()

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


    def calc_markers_base_position(self, corners, ids):
        if len(corners) == 0:
            return []
        rvecs, tvecs = solve_marker_pnp(corners, self.marker_size, self.mtx, self.dist)  
        for i, tvec, rvec in zip(ids, tvecs, rvecs):
            tvec = tvec.squeeze().tolist()
            rvec = rvec.squeeze().tolist()
            rotvector = np.array([[rvec[0], rvec[1], rvec[2]]])
            Rotation = cv2.Rodrigues(rotvector)[0]  
            Euler = self.CvtRotationMatrixToEulerAngle(Rotation) 
            target_coords = np.array([tvec[0], tvec[1], tvec[2], Euler[0], Euler[1], Euler[2]])
        return target_coords

    def stag_robot_identify(self, ml):
        marker_pos_pack,ids = self.stag_identify()
        target_coords = self.safe_get_coords(ml,10,0.5)
        while (target_coords is None):
            target_coords = self.safe_get_coords(ml,10,0.5)
        # print("current_coords", target_coords)
        cur_coords = np.array(target_coords.copy())
        cur_coords[-3:] *= (np.pi / 180) 
        fact_bcl = self.Eyes_in_hand(cur_coords, marker_pos_pack, self.EyesInHand_matrix)
        
        for i in range(3):
            target_coords[i] = fact_bcl[i]
        
        return target_coords,ids
    
    
    def CvtRotationMatrixToEulerAngle(self, pdtRotationMatrix):
        pdtEulerAngle = np.zeros(3)
        pdtEulerAngle[2] = np.arctan2(pdtRotationMatrix[1, 0], pdtRotationMatrix[0, 0])
        fCosRoll = np.cos(pdtEulerAngle[2])
        fSinRoll = np.sin(pdtEulerAngle[2])
        pdtEulerAngle[1] = np.arctan2(-pdtRotationMatrix[2, 0],
                                      (fCosRoll * pdtRotationMatrix[0, 0]) + (fSinRoll * pdtRotationMatrix[1, 0]))
        pdtEulerAngle[0] = np.arctan2((fSinRoll * pdtRotationMatrix[0, 2]) - (fCosRoll * pdtRotationMatrix[1, 2]),
                                      (-fSinRoll * pdtRotationMatrix[0, 1]) + (fCosRoll * pdtRotationMatrix[1, 1]))
        return pdtEulerAngle

    def CvtEulerAngleToRotationMatrix(self, ptrEulerAngle):
        ptrSinAngle = np.sin(ptrEulerAngle)
        ptrCosAngle = np.cos(ptrEulerAngle)
        ptrRotationMatrix = np.zeros((3, 3))
        ptrRotationMatrix[0, 0] = ptrCosAngle[2] * ptrCosAngle[1]
        ptrRotationMatrix[0, 1] = ptrCosAngle[2] * ptrSinAngle[1] * ptrSinAngle[0] - ptrSinAngle[2] * ptrCosAngle[0]
        ptrRotationMatrix[0, 2] = ptrCosAngle[2] * ptrSinAngle[1] * ptrCosAngle[0] + ptrSinAngle[2] * ptrSinAngle[0]
        ptrRotationMatrix[1, 0] = ptrSinAngle[2] * ptrCosAngle[1]
        ptrRotationMatrix[1, 1] = ptrSinAngle[2] * ptrSinAngle[1] * ptrSinAngle[0] + ptrCosAngle[2] * ptrCosAngle[0]
        ptrRotationMatrix[1, 2] = ptrSinAngle[2] * ptrSinAngle[1] * ptrCosAngle[0] - ptrCosAngle[2] * ptrSinAngle[0]
        ptrRotationMatrix[2, 0] = -ptrSinAngle[1]
        ptrRotationMatrix[2, 1] = ptrCosAngle[1] * ptrSinAngle[0]
        ptrRotationMatrix[2, 2] = ptrCosAngle[1] * ptrCosAngle[0]
        return ptrRotationMatrix
    
    def eyes_in_hand_calculate(self, pose, tbe, Mc, Mr):
        pose,Mr =  map(np.array, [pose,Mr])
        euler = pose * np.pi / 180
        Rbe = self.CvtEulerAngleToRotationMatrix(euler)
        Reb = Rbe.T

        A = np.empty((3, 0))
        b_comb = np.empty((3, 0))
        
        r = tbe.shape[0]
        
        for i in range(1, r):
            A = np.hstack((A, (Mc[i, :].reshape(3, 1) - Mc[0, :].reshape(3, 1))))
            b_comb = np.hstack((b_comb, (tbe[0, :].reshape(3, 1) - tbe[i, :].reshape(3, 1))))
        
        b = Reb @ b_comb
        U, _, Vt = svd(A @ b.T)
        Rce = Vt.T @ U.T
        
        tbe_sum = np.sum(tbe, axis=0)
        Mc_sum = np.sum(Mc, axis=0)
        
        tce = Reb @ (Mr.reshape(3, 1) - (1/r) * tbe_sum.reshape(3, 1) - (1/r) * (Rbe @ Rce @ Mc_sum.reshape(3, 1)))
        tce[2] -= self.IDENTIFY_LEN

        EyesInHand_matrix = np.vstack((np.hstack((Rce, tce)), np.array([0, 0, 0, 1])))
        print("EyesInHand_matrix = ", EyesInHand_matrix)
        return EyesInHand_matrix
    
    def Transformation_matrix(self,coord):
        position_robot = coord[:3]
        pose_robot = coord[3:]
        RBT = self.CvtEulerAngleToRotationMatrix(pose_robot) 
        PBT = np.array([[position_robot[0]],
                        [position_robot[1]],
                        [position_robot[2]]])
        temp = np.concatenate((RBT, PBT), axis=1)
        array_1x4 = np.array([[0, 0, 0, 1]])
        matrix = np.concatenate((temp, array_1x4), axis=0) 
        return matrix

    def Eyes_in_hand(self, coord, camera, Matrix_TC):
        Position_Camera = np.transpose(camera[:3])  
        Matrix_BT = self.Transformation_matrix(coord) 

        Position_Camera = np.append(Position_Camera, 1)  
        Position_B = Matrix_BT @ Matrix_TC @ Position_Camera
        return Position_B


    
    def stag_identify(self):
        self.camera.update_frame()  
        frame = self.camera.color_frame()  
        (corners, ids, rejected_corners) = stag.detectMarkers(frame, 11)  
        marker_pos_pack = self.calc_markers_base_position(corners, ids) 
        if(len(marker_pos_pack) == 0):
            marker_pos_pack, ids = self.stag_identify()
    
        return marker_pos_pack, ids

    def Matrix_identify(self, ml):
        ml.send_angles(self.origin_mycbot_level, 50)  
        self.wait()
        input("make sure camera can observe the stag, enter any key quit")
        coords = self.safe_get_coords(ml,10,0.5)
        pose = coords[3:6]
        print(pose)
        # self.camera_open_loop()
        Mc1,tbe1,pos1 = self.reg_get(ml)
        ml.send_coord(1, coords[0] + 50, 30)
        self.wait()
        Mc2,tbe2,pos2 = self.reg_get(ml)
        ml.send_coord(3, coords[2] + 20, 30)
        self.wait()
        Mc3,tbe3,pos3 = self.reg_get(ml)
        ml.send_coord(2, coords[1] + 20, 30)
        self.wait()
        Mc4,tbe4,pos4 = self.reg_get(ml)
        ml.send_coord(1, coords[0] + 20, 30)
        self.wait()
        Mc5,tbe5,pos5 = self.reg_get(ml)
        tbe = np.vstack([tbe1, tbe2, tbe3, tbe4, tbe5])
        Mc = np.vstack([Mc1, Mc2, Mc3, Mc4, Mc5])
        state = None
        if self.EyesInHand_matrix is not None:
            state = True
            pos = np.vstack([pos1, pos2, pos3, pos4, pos5])
            r = pos.shape[0]
            for i in range(1, r):
                for j in range(3):
                    err = abs(pos[i][j] - pos[0][j])
                    if(err > 10):
                        state = False
                        print("matrix error")
        return pose, tbe, Mc, state

    def Eyes_in_hand_calibration(self, ml):
        mc.set_end_type(0)
        pose, tbe, Mc, state = self.Matrix_identify(ml)
        if(state == True):
            print("Calibration Complete EyesInHand_matrix = ", self.EyesInHand_matrix)
            return

        input("Move the end of the robot arm to the calibration point, press any key to release servo")
        ml.release_all_servos()
        input("focus servo and get current coords")
        ml.power_on()
        time.sleep(1)
        coords = self.safe_get_coords(ml,10,0.5)
        while len(coords) == 0:
            coords = self.safe_get_coords(ml,10,0.5)
        Mr = coords[0:3]
        print(Mr)


        self.EyesInHand_matrix = self.eyes_in_hand_calculate(pose, tbe, Mc, Mr)
        print("EyesInHand_matrix = ", self.EyesInHand_matrix)
        self.save_matrix()  # Save the matrix to a file after calculating it
        print("save successe, wait to verify")

        pose, tbe, Mc, state = self.Matrix_identify(ml)
        if state != True:
            self.EyesInHand_matrix = self.eyes_in_hand_calculate(pose, tbe, Mc, Mr)

    def reg_get(self, ml):
        target_coords = None
        for i in range(30):
            Mc_all,_ = self.stag_identify()
        if self.EyesInHand_matrix is not None:
            target_coords,_ = self.stag_robot_identify(ml)

        tbe_all = self.safe_get_coords(ml,10,0.5) 
        while (tbe_all is None):
            tbe_all = self.safe_get_coords(ml,10,0.5)

        tbe = np.array(tbe_all[0:3])
        Mc = np.array(Mc_all[0:3])
        print("tbe = ", tbe)
        print("Mc = ", Mc)
        return Mc,tbe,target_coords

if __name__ == "__main__":
    camera_params = np.load("camera_params.npz") 
    mtx, dist = camera_params["mtx"], camera_params["dist"]
    m = CameraCalibrate(0, 85, mtx, dist)
    mc.set_vision_mode(1)
    m.Eyes_in_hand_calibration(mc)

