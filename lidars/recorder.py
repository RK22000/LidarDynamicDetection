import numpy as np
import time
import sys
import re
# sys.path.append((next(re.finditer(".*unified_frameworks", __file__)).group()))
from LidarClass import _Lidar
from actual_lidar import ActualLidar
from utilf import Service, polar_sum, polar_to_cart, cart_to_polar
import json

config = {
    "verbose":False,
    "port": "COM8"
}
class RecorderLidar(ActualLidar):
    def __init__(self, port) -> None:
        super().__init__(port)
        self.recording = {}
        
        def update(is_alive):
            while is_alive():
                measures = self.get_measures()
                self.recording[time.time()] = {
                    "quality": measures[:,0].tolist(),
                    "degrees": measures[:,1].tolist(),
                    "milimeters": measures[:,2].tolist()
                }
                time.sleep(1/25)
        self.recorder = Service(update, "Recorder thread")
    def start_recording(self):
        self.recorder.start_service()
    def stop_recording(self):
        self.recorder.stop_service()
    def save_recording(self, file):
        with open(file, 'w') as f:
            json.dump(self.recording, f, indent=2)

if __name__=='__main__':
    RecorderLidar().test_Lidar()
