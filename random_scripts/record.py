import os, sys
sys.path.append(os.path.realpath("."))
import matplotlib.pyplot as plt
import numpy as np

from lidars.recorder import RecorderLidar
lidar = RecorderLidar("COM8")
if not lidar.connect(): exit(1)
try:
    ax = plt.subplot(projection='polar')
    plt.show(block=False)
    lidar.start_recording()
    while plt.get_fignums():
        ax.clear()
        polar_points = lidar.get_measures()[:, [1,2]]*[np.pi/180,1/1000]
        ax.scatter(*(polar_points).T, s=1, c='k')
        plt.pause(0.1)
    lidar.stop_recording()
    lidar.save_recording("Recording2.json")
except:
    pass
lidar.disconnect()