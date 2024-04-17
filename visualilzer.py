import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.axes import Axes
import numpy as np


class Visualizer:
    """
    A visualizer to visualize a bunch of points and lines
    """
    def __init__(self, ax:Axes=None) -> None:
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
            ax:Axes=ax
        self.points = ax.scatter([],[],s=1)
        ax.set_rmax(50)
        self.lc = LineCollection([],)
        ax.add_collection(self.lc)


        self.ax = ax
    
    def set_points(self, points: list[list[float]], sizes=1):
        """
        Set points on the visualizer
        
        :param points: list of 2d points
        
        """
        self.points.set_offsets(points)
        if not hasattr(sizes, '__len__'): sizes = [sizes]*len(points)
        self.points.set_sizes(sizes)
        rate=0.9
        # self.ax.set_rmax(rate*(1.3*max([d for a,d in points]))+(1-rate)*self.ax.get_rmax())
        self.adjust_to_r(max([d for a,d in self.get_all_points()]))
        plt.pause(0.1)
    
    def get_all_points(self):
        """Get a list of all points (even the ones in lines) being visualized"""
        segments = [seg.tolist() for seg in self.lc.get_segments()]
        return sum(segments, []) + self.points.get_offsets().tolist()
    
    def adjust_to_r(self, r):
        rate=0.9
        self.ax.set_rmax(rate*(1.3*r)+(1-rate)*self.ax.get_rmax())

    def set_lines(self, segments: list[list[list[float]]]):
        """
        Set lines on the visualizer
        
        :param segments: A list of lines, where each line is a list of points

        """
        self.lc.set_segments(segments)
        # r=max(sum([[d for a,d in seg] for seg in segments],[]))
        r = max([d for a,d in self.get_all_points()])
        self.adjust_to_r(r)
        plt.pause(0.1)
    
    def cartify_polar_points(self, polar_points: list[list[float]]):
        """Convert cartesian points to polar points"""
        def cartify_point(p):
            a,d = p
            return (d*np.cos(a), d*np.sin(a))
        return [cartify_point(p) for p in polar_points]
    def polarize_cart_points(self, cart_points: list[list[float]]):
        """Convert cartesian points to polar points"""
        return [[np.arctan2(y,x),np.linalg.norm((x,y))] for x,y in cart_points]