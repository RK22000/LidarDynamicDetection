"""
This is a stepometer. Given two consecutive 2d point clouds, it attempts to 
calculate the step taken by which explains the difference in the two scans 

"""
import tensorflow as tf
import numpy as np
from heapq import nsmallest

class Stepometer(tf.Module):
    def __init__(self, name=None, copy=None):
        super().__init__(name)
        self.pos =   tf.Variable(tf.zeros(2), name="DeltaPosition")
        self.theta = tf.Variable(tf.zeros(1), name="DeltaTheta")
        # self.theta = 0
    def reset(self):
        self.pos.assign(tf.zeros(2))
        self.theta.assign(tf.zeros(1))

    @property
    def ROTATION_MATRIX(self):
        r0 = tf.stack([tf.cos(self.theta), -tf.sin(self.theta)], axis=1)
        r1 = tf.stack([tf.sin(self.theta),  tf.cos(self.theta)], axis=1)
        return tf.concat([r0, r1], axis=0)
    @property
    def BACK_ROTATION_MATRIX(self):
        r0 = tf.stack([tf.cos(-self.theta), -tf.sin(-self.theta)], axis=1)
        r1 = tf.stack([tf.sin(-self.theta),  tf.cos(-self.theta)], axis=1)
        return tf.concat([r0, r1], axis=0)
    def __call__(self, points):
        # if not hasattr(points, "T"): points = np.array(points)
        return tf.transpose(self.ROTATION_MATRIX @ (points).T) + self.pos
    def revert(self, points):
        return tf.transpose(self.BACK_ROTATION_MATRIX @ tf.transpose(points-self.pos))

    def fit_once(self, p0, p1, lr=0.1, well_width=0.1):
        with tf.GradientTape() as tape:
            # errors = tf.stack([p1-p for p in self(p0)])
            # squared = errors * errors
            # exponentiated = tf.exp(-squared/0.9)
            # xy_combined = tf.reduce_prod(exponentiated, axis=-1)
            # unoccupied_probs = 1-xy_combined
            # losses = tf.reduce_prod(unoccupied_probs, axis=-1) + 0.1*(tf.reduce_sum(self.pos*self.pos) + self.theta*self.theta)
            losses = self.calc_loss(p0, p1, lr, well_width)
            loss = sum(losses)/len(losses)
        [d_pos, d_theta] = tape.gradient(loss, [self.pos, self.theta])
        self.pos.assign_sub(lr*d_pos)
        self.theta.assign_sub(lr*0.1*d_theta)
        return loss
    def calc_loss(self, p0, p1, lr=0.001, well_width=0.1, forward=True):
        errors = tf.stack([p1-p for p in {True:self.__call__, False:self.revert}[forward](p0)])
        squared = errors * errors
        squared /= ([[[{True:1,False:np.linalg.norm(p)}[forward]]*2]*len(p1) for p in p0])
        exponentiated = tf.exp(-squared/well_width)
        xy_combined = tf.reduce_prod(exponentiated, axis=-1)
        unoccupied_probs = 1-xy_combined
        losses = tf.reduce_prod(unoccupied_probs, axis=-1) + 0.01*(tf.reduce_sum(self.pos*self.pos))# + self.theta*self.theta)
        return losses
    def heuristic_fit(self, p0, p1):
        """
        Fit to the step using a heuristic
        
        """
        p,th1 = heuristic_pos_theta(p0, p1)
        self.pos.assign(p)
        self.theta.assign(np.array([th1]))
    def __repr__(self) -> str:
        return f"Stepometer<pos: {self.pos.numpy()}, theta: {self.theta.numpy()}>"

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
