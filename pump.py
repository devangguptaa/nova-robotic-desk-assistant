from pymycobot import MyCobot280
import time
# init_angles=[33.22, -15.55, -30.54, 25.48, 6.76, -13.35]#6 joint angles at the initial position
# grab_point=[189.9, 12.1, 84.5, -178.15, -96.89, -43.47]#Coordinates of the grab point
# # grab_point = [189.9, 12.1, 84.5, -178.15, 6.89, 46.53]

# place_point=[189.9, 120.1, 62.5, -178.15, 6.89, -43.47]# Coordinates of the placement point

# mc = MyCobot280("/dev/tty.usbmodem1101",115200)
# time.sleep(2)
# # Turn on the suction pump

class PumpControl:
    def __init__(self, mc):
        self.mc = mc
    def pump_on(self):
        print("on")
        self.mc.set_digital_output(33, 0)
        time.sleep(0.05)

    # Stop the pump
    def pump_off(self):
        print("off")
        self.mc.set_digital_output(33, 1)
        time.sleep(0.05)
        self.mc.set_digital_output(23, 0)
        time.sleep(1)
        self.mc.set_digital_output(23, 1)
        time.sleep(0.05)