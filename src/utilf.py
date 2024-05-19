import numpy as np
from abc import ABC, abstractmethod
from threading import Thread

def polar_to_cart(p):
    a,d = p
    return d*np.array([np.cos(a),np.sin(a)])
def cart_to_polar(p):
    x,y=p
    return np.array([np.arctan2(y,x), np.linalg.norm(p)])
def make_polar_points(n=150):
    r = np.random.randn(n)
    d0 = r*1+6
    d0 = np.convolve(d0, [1/(int(0.03*n))]*int(0.03*n), mode='same')
    a0 = np.linspace(0, 2*np.pi, n)

    points = np.stack([a0,d0], axis=1)
    points = np.array([polar_to_cart(p) for p in points])
    points *= (np.random.randn(2)+1)
    points = np.array([cart_to_polar(p) for p in points])

    return points

def heuristic_pos_theta(points0, points1):
    '''Takes 2d list of cartesian points
    return position, theta, -theta%2pi
    '''
    pos0 = np.sum(points0, 0)/len(points0)
    cov = np.cov(points0.T)
    eval, evec = np.linalg.eig(cov)
    p0 = evec[:, np.argmax(eval)]
    t0 = np.arctan2(*p0[::-1]) + 3*np.pi/4

    pos1 = np.sum(points1, 0)/len(points1)
    cov = np.cov(points1.T)
    eval, evec = np.linalg.eig(cov)
    p1 = evec[:, np.argmax(eval)]
    t1 = np.arctan2(*p1[::-1]) + 3*np.pi/4

    delta_thetas = np.array([t1-t0, t1-t0+np.pi, t1-t0-np.pi])
    theta = delta_thetas[np.argmin(np.abs(delta_thetas))]
    pos0_ = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ]) @ np.reshape(pos0, (2,-1))
    pos0_ = np.reshape(pos0_, (2))
    pos = pos1-pos0_
    return (pos, theta)

class _Abstract_Service(ABC):
    @abstractmethod
    def start_service(self):
        pass
    @abstractmethod
    def stop_service(self):
        pass

class Service(_Abstract_Service):
    def __init__(self, service_func, name) -> None:
        """
        Create an easy to start/stop service from a function. 
        The function must accept a callable which returns if the service is still running. 
        That callable must be respected to prevent the service from running amok.
        """
        self._running = False
        self._service_func = service_func
        self._thread = Thread(target=service_func, args=(lambda: self._running,), name=name)
    def start_service(self):
        self._running=True
        self._thread.start()
    def stop_service(self):
        self._running=False
        self._thread.join()
        self._thread = Thread(target=self._service_func, args=(lambda: self._running,), name=self._thread.name)
    def is_running(self):
        return self._running

from math import sqrt, cos, sin, atan2
def polar_dis(p1, p2):
    """Distance between polar coordinates p1 and p2"""
    if p1 is None or p2 is None: return float('inf')
    return sqrt(abs(p1[1]**2 + p2[1]**2 - 2*p1[1]*p2[1]*cos(p1[0]-p2[0])))
def same_polar_point(p1, p2, thresh=0.1):
    # return False
    return polar_dis(p1, p2) < thresh
def polar_to_cart(p) -> np.ndarray:
    if p is None: return None
    return p[1]*np.array([cos(p[0]), sin(p[0])])
def cart_to_polar(coord):
    if coord is None: return None
    x, y = coord
    res = np.array([atan2(y, x), np.linalg.norm(coord)])
    # print(f"cart_to_polar: {res}")
    return res
def polar_sum(*polar_points):
    cart_points = [polar_to_cart(p) for p in polar_points if p is not None]
    return cart_to_polar(sum(cart_points, np.array([0,0])))

def normalize_cart_scan(scan, thresh_dis):
    """Even out the spacing of points in a point cloud scan by dropping points 
    if they are too near
    The points have to be (x meters, y meters)
    
    """
    polar_scan = [cart_to_polar(p) for p in scan]
    normal_polar = normalize_polar_scan(polar_scan, thresh_dis)
    normal_cart = [polar_to_cart(p) for p in normal_polar]
    return np.array(normal_cart)

def normalize_polar_scan(scan, thresh_dis):
    """Even out the spacing of points in a point cloud scan by dropping points 
    if they are too near
    The points have to be (radians, meters)
    
    """
    measures = [(rad%(2*np.pi), met) for rad, met in scan]
    measures = (sorted(measures))
    normed = [measures[0]]
    for a,d in measures:
        p0 = np.array(normed[-1]) 
        p1 = np.array([a,d])          
        if polar_dis(p0, p1) > thresh_dis:
        # if False:
            normed.append([a,d])

    return np.array(normed)