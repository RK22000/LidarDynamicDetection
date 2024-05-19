print("Starting imports")
import json
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.realpath("."))

from lidars import normalized_lidar, recording_lidar, fake_lidar, actual_lidar
from src import utilf, tf_stepometer
print("Finished imports")

# lidar = actual_lidar.ActualLidar("COM8")
# lidar = recording_lidar.RecordingLidar(f"lidars{os.sep}labrecfix.json")
# lidar = recording_lidar.RecordingLidar(f"Recording2.json", 1, loop=False)
# lidar = normalized_lidar.NormalizedLidar(lidar, 0.4)
lidar = fake_lidar.FakeLidar()
if not lidar.connect():
    print("Failed to connect to lidar")
    import sys
    sys.exit(1)
s = tf_stepometer.Stepometer()

try:
    print("Creating Figure")
    plt.figure()
    print("Creating subplot")
    ax = plt.subplot(projection='polar')
    print("Showing")
    # plt.show(block=False)
    r=None
    size_lag = 0.9
    prev_points = None
    cross_hair = np.array([(2,0),(0,0),(0,2)])
    travel_path = np.array([(0,0)])
    while plt.get_fignums():
        ax.clear()
        polar_points = lidar.get_measures()[:, [1,2]]*[np.pi/180,1/1000]
        ax.scatter(*(polar_points).T, s=1, c='k')
        cart_points = np.array([utilf.polar_to_cart(p) for p in polar_points])
        
        # Center of Mass and Principal Component basis
        com = np.sum(cart_points, 0)/len(cart_points)
        ax.scatter(*utilf.cart_to_polar(com), c='r')
        cov = np.cov(cart_points.T)
        eval, evec = np.linalg.eig(cov)
        basis = np.stack([np.zeros_like(evec.T), evec.T], 1).reshape(-1,2)
        basis += com
        basis = np.array([utilf.cart_to_polar(p) for p in basis])
        ax.plot(*basis[:2].reshape(2,2).T)
        ax.plot(*basis[2:].reshape(2,2).T)

        if prev_points is None: prev_points = cart_points
        # normalized_prev = utilf.normalize_cart_scan(prev_points, 0.3)
        # indexes = np.random.randint(0, len(normalized_prev), (10, 6))
        # well_widths = np.linspace(1, 0, len(indexes))**5 * 0 + 0.1
        # for idx, ww in zip(indexes, well_widths): loss = s.fit_once(normalized_prev[idx],cart_points, well_width=ww)
        # motion_losses = s.calc_loss(cart_points, prev_points, forward=False)
        # x = ax.scatter(*(polar_points).T, s=5, c=motion_losses, cmap='plasma')
        s.heuristic_fit(prev_points, cart_points)
        x = ax.scatter(*(polar_points).T, s=1)
        # try:cb.remove()
        # except:NameError
        # print(x)


        # Training points overlay
            # train_points = normalized_prev[np.unique(indexes.flatten())]
            # train_points = s(train_points).numpy()
            # train_points = [utilf.cart_to_polar(p) for p in train_points]
            # ax.scatter(*zip(*train_points), s=2, c='r')
        
        # Set new Prev points
        prev_points = cart_points

        # Cross hairs
            # cross_hair = s(cross_hair).numpy()
            # ax.plot(*zip(*[utilf.cart_to_polar(p) for p in cross_hair]))

        travel_path = np.concatenate([s(travel_path[-50:]), [(0,0)]], 0)
        ax.plot(*zip(*[utilf.cart_to_polar(p) for p in travel_path]), color='m')





        if r is None: r=max(polar_points[:,1])
        ax.set_rmax(size_lag*r+(1-size_lag)*(max(polar_points[:,1])+2))
        # ax.set_rmax(10)
        r = ax.get_rmax()

        plt.pause(0.1)
except Exception as e:
    import traceback
    traceback.print_exc()

lidar.disconnect()