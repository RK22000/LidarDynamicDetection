"""This lidar will act as a filter on top of another lidar"""

from lidars.LidarClass import _Lidar
import numpy as np
from utilf import polar_dis

class NormalizedLidar(_Lidar):
    def __init__(self, lidar: _Lidar, thresh=0.2) -> None:
        self._lidar = lidar
        self.thresh=thresh
        pass
    def connect(self, max_attempts=3, wait_seconds=1, verbose_attempts=True) -> bool:
        return self._lidar.connect(max_attempts, wait_seconds, verbose_attempts)
    def get_measures(self):
        measures = self._lidar.get_measures()
        measures = (sorted(measures, key=lambda i: i[1]))
        normed = [measures[0]]
        for q,a,d in measures[1:]:
            p0 = np.array(normed[-1][1:]) * [np.pi/180, 1/1000]
            p1 = np.array([a,d])          * [np.pi/180, 1/1000]
            if polar_dis(p0, p1) > self.thresh:
            # if False:
                normed.append([q,a,d])

        return np.array(normed)

    def disconnect(self):
        self._lidar.disconnect()
        pass