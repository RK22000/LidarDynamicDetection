import numpy as np
import time
import sys
import re
# sys.path.append((next(re.finditer(".*unified_frameworks", __file__)).group()))
from LidarClass import _Lidar
from utilf import Service, polar_sum, polar_to_cart, cart_to_polar
import json

config = {
    "verbose":False
}
class RecordingLidar(_Lidar):
    class RecordingEndedException(Exception):
        pass
    def __init__(self, recording_file, speed_multiplier=1, loop=True) -> None:
        with open(recording_file, 'r') as f:
            recording = json.load(f)
            self.recording = {float(k): recording[k] for k in recording}
        keys = (self.recording.keys())
        self._ts0 = min(keys)
        self._tsn = max(keys)
        self.speed_multiplier = speed_multiplier
        self.loop = loop
        pass
    def connect(self, max_attempts=3, wait_seconds=1, verbose_attempts=True) -> bool:
        self._tsStart = time.time()
        return True
    def get_measures(self):
        ts = (self.speed_multiplier*(time.time()-self._tsStart))
        if ts > (self._tsn-self._ts0) and not self.loop:
            raise RecordingLidar.RecordingEndedException()
        ts = ts%(self._tsn-self._ts0)

        ts = min(self.recording.keys(), key=lambda k: abs(ts-(k-self._ts0)))
        quality, degrees, millimeters = self.recording[ts].values()
        return np.stack([quality, degrees, millimeters], 1)

    def disconnect(self):
        pass

if __name__=='__main__':
    RecordingLidar("labrecfix.json").test_Lidar()
