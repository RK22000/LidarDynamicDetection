import os, sys
sys.path.append(os.path.realpath("."))
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import argparse
from src import utilf
import threading

parser = argparse.ArgumentParser(description="Launch the visualization with or without the stepmother")
parser.add_argument("-s", "--Stepometer", action="store_true", help="Run the stepometer to track path")
parser.add_argument('-f', default="Recording2.json", help="The recording file to use")
parser.add_argument("-l", action="store_true", help="Connect to a live Lidar")
parser.add_argument("-m", action="store_true", help="Detect dynamic motion in scans")
args = parser.parse_args()
if args.Stepometer:
    import time
    from src import tf_stepometer
print(args)

from lidars.recording_lidar import RecordingLidar
from lidars.actual_lidar import ActualLidar
# lidar = RecorderLidar("COM8")
if args.l:
    lidar = ActualLidar("COM8")
else:
    lidar = RecordingLidar(args.f)
if not lidar.connect(): exit(1)
stuff = {}
if args.Stepometer:
    s = tf_stepometer.Stepometer()
    def motion_tracker(cart_points, prev_points):
        # cart_points = utilf.normalize_cart_scan(cart_points,0.3)
        # cart_points = cart_points[
        #         np.random.randint(0,len(cart_points), len(cart_points)//10)
        #     ]
        backlosses = s.calc_loss(cart_points, prev_points, forward=False) / np.power(np.linalg.norm(cart_points, axis=1), 2)
        topPercentile = np.percentile(backlosses, 92)
        stuff["motion"] = cart_points[backlosses>topPercentile]

    def path_tracker(is_tracking):
        travel_path = np.array([(0,0)])
        while is_tracking():
            polar_points = lidar.get_measures()[:, [1,2]]*[np.pi/180, 1/1000]
            try: prev_points = cart_points
            except NameError: pass
            cart_points = np.array([utilf.polar_to_cart(p) for p in polar_points])
            try: prev_points
            except NameError: prev_points = cart_points
            # pos = s.pos.numpy()
            # s.heuristic_fit(prev_points, cart_points)
            # s.pos.assign(pos)
            iters = 5
            prev_samples = prev_points[
                np.random.randint(0,len(prev_points), (iters, 10))
            ]
            for sample in prev_samples: loss = s.fit_once(sample, cart_points, lr=0.1)
            # print(loss)
            travel_path = np.concatenate([s(travel_path[:]), [(0,0)]], 0)
            stuff['travel_path']=travel_path
            if args.m and np.random.rand()<0.1:
                t = threading.Thread(target=motion_tracker, args=(cart_points,prev_points))
                t.start()
    stepper_service = utilf.Service(path_tracker, "Stepometer Service")

try: stepper_service.start_service()
except NameError: print("failed to start steper")

try:
    ax = plt.subplot(projection='polar')
    plt.show(block=False)
    while plt.get_fignums():
        ax.clear()
        polar_points = lidar.get_measures()[:, [1,2]]*[np.pi/180,1/1000]
        ax.scatter(*(polar_points).T, s=1, c='k')
        if 'travel_path' in stuff:
            ax.plot(*zip(*[utilf.cart_to_polar(p) for p in stuff['travel_path']]), color='m')
        if 'motion' in stuff and stuff['motion'].shape[0]>0:
            # print(stuff['motion'].shape)
            x, y = zip(*[utilf.cart_to_polar(p) for p in stuff['motion']])
            ax.scatter(x, y, color='r', s=5)



        try: r
        except NameError: r = max(polar_points[:,1])
        lag = 0.99
        ax.set_rmax(lag*r+(1-lag)*(max(polar_points[:,1])+2))
        r = ax.get_rmax()
        plt.pause(0.1)
except:
    import traceback
    traceback.print_exc()
    pass
try: stepper_service.stop_service()
except NameError: pass

lidar.disconnect()