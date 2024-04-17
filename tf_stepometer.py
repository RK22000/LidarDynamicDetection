"""
This is a stepometer. Given two consecutive 2d point clouds, it attempts to 
calculate the step taken by which explains the difference in the two scans 

"""
import tensorflow as tf
import numpy as np
from heapq import nsmallest

class Stepometer(tf.Module):
    def __init__(self, name=None):
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

    def fit_once(self, p0, p1, lr=0.001, optim=True):
        with tf.GradientTape() as tape:
            # errors = tf.stack([p1-p for p in self(p0)])
            # squared = errors * errors
            # exponentiated = tf.exp(-squared/0.9)
            # xy_combined = tf.reduce_prod(exponentiated, axis=-1)
            # unoccupied_probs = 1-xy_combined
            # losses = tf.reduce_prod(unoccupied_probs, axis=-1) + 0.1*(tf.reduce_sum(self.pos*self.pos) + self.theta*self.theta)
            losses = self.calc_loss(p0, p1, lr)
        [d_pos, d_theta] = tape.gradient(losses, [self.pos, self.theta])
        self.pos.assign_sub(lr*d_pos)
        self.theta.assign_sub(lr*d_theta)
        return sum(losses)
    def calc_loss(self, p0, p1, lr=0.001):
        errors = tf.stack([p1-p for p in self(p0)])
        squared = errors * errors
        exponentiated = tf.exp(-squared/0.9)
        xy_combined = tf.reduce_prod(exponentiated, axis=-1)
        unoccupied_probs = 1-xy_combined
        losses = tf.reduce_prod(unoccupied_probs, axis=-1) + 0.1*(tf.reduce_sum(self.pos*self.pos) + self.theta*self.theta)
        return losses