"""
This file is meant to be a fake lidar script that should be a 
drop in replacement to the actual RPLidar
I think making this will help in testing an optimizing the lidar scripts
"""
import numpy as np
import time
import sys
import re
# sys.path.append((next(re.finditer(".*unified_frameworks", __file__)).group()))
from lidars.LidarClass import _Lidar
from utilf import Service, polar_sum, polar_to_cart, cart_to_polar

config = {
    "verbose":False
}
class FakeLidar(_Lidar):
    def __init__(self, points=100, angular_rate=1, translational_rate=5, jitter=0, noise=0, empty_scans=False, refreshHZ=10) -> None:
        self.n = points
        angles = np.linspace(0, 360, self.n)
        distances = [5_000]
        while len(distances) < self.n:
            shift = 1 # if np.random.rand() < 1 else 1
            d = (distances[-1]+(np.random.randn()*shift*1_000))/1
            if d < 2_000: d*=1.3
            distances.append(d)#istances[-1]+(np.random.randn()*shift*1_000))
        distances = np.convolve(distances, [1/(int(0.03*self.n))]*int(0.03*self.n), 'same')
        quality = [15]*self.n
        self.scan = np.stack([quality, angles, distances], 1)
        # self.scan = np.random.rand(self.n, 3)*1_000 +2_000
        self.empty_scans = empty_scans
        self.noise = noise
        theta = angular_rate/refreshHZ
        t = np.random.rand()*2*np.pi
        pos = np.array([np.cos(t), np.sin(t)])*1000*translational_rate/refreshHZ
        def update(is_alive):
            while is_alive():
                theta_ = theta + np.random.randn()*jitter
                pos_ = pos + np.random.randn(*pos.shape)*jitter/10

                points = np.array([polar_to_cart(p*[np.pi/180,1]) for p in self.scan[:, [1,2]]])
                points = (np.array([
                    [np.cos(theta_), -np.sin(theta_)],
                    [np.sin(theta_),  np.cos(theta_)]
                ]) @ points.T).T + pos_
                self.scan[:, [1,2]] = [cart_to_polar(p)*[180/np.pi,1] for p in points]
                time.sleep(1/refreshHZ)
        self.s = Service(update, "Fake Lidar Generator")
        pass
    def connect(self, max_attempts=3, wait_seconds=1, verbose_attempts=True) -> bool:
        self.s.start_service()
        return True
    def get_measures(self):
        if self.empty_scans:
            return []
        # self.scan += (0,1,0)
        res = self.scan + (np.random.randn(self.n, 3))*(0,0,100)*self.noise
        print(res) if config["verbose"] else None
        return res
        # return self.scan
    def disconnect(self):
        self.s.stop_service()
        pass

if __name__=='__main__':
    FakeLidar().test_Lidar()