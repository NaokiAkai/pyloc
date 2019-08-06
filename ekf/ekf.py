'''
Copyright (C) 2019 Naoki Akai.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at https://mozilla.org/MPL/2.0/.
'''

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import copy
from numpy.random import normal, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat

class EKF:
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.robot_pose_x = x
        self.robot_pose_y = y
        self.robot_pose_yaw = yaw
        self.pose_cov = np.eye(3)

        self.odom_noise1 = 1.0
        self.odom_noise2 = 0.5
        self.odom_noise3 = 0.5
        self.odom_noise4 = 2.0
        self.range_variance = 0.1 * 0.1
        self.angle_variance = 0.01 * 0.01
        self.Q = np.array([[self.range_variance, 0.0],
                           [0.0, self.angle_variance]])
        self.min_trace = 0.01
        self.landmarks = []

        self.plot_size_x = 5.0
        self.plot_size_y = 5.0

        self.PI = 3.14159265359
        self.PI2 = 6.28318530718

    def mod_yaw(self, yaw):
        while yaw < -self.PI:
            yaw += self.PI2
        while yaw > self.PI:
            yaw -= self.PI2
        return yaw

    def set_odom_noises(self, odom_noise1, odom_noise2, odom_noise3, odom_noise4):
        self.odom_noise1 = odom_noise1
        self.odom_noise2 = odom_noise2
        self.odom_noise3 = odom_noise3
        self.odom_noise4 = odom_noise4

    def set_min_trace(self, min_trace):
        self.min_trace = min_trace

    def set_plot_sizes(self, plot_size_x, plot_size_y):
        self.plot_size_x = plot_size_x
        self.plot_size_y = plot_size_y

    def add_landmark(self, x, y):
        self.landmarks.append([x, y])

    def print_landmarks(self):
        for i in range(len(self.landmarks)):
            print i, 'th landmark: x =', self.landmarks[i][0], '[m], y =', self.landmarks[i][1], ' [m]'
        print ''

    def print_estimated_pose(self):
        print 'x =', self.robot_pose_x, '[m], y =', self.robot_pose_y, '[m], yaw =', self.robot_pose_yaw * 180.0 / self.PI, '[deg]'

    def print_estimated_covariance(self):
        print 'covariance\n', self.pose_cov

    def predict(self, delta_dist, delta_yaw):
        # update the robot pose
        x = self.robot_pose_x + delta_dist * math.cos(self.robot_pose_yaw)
        y = self.robot_pose_y + delta_dist * math.sin(self.robot_pose_yaw)
        yaw = self.robot_pose_yaw + delta_yaw
        self.robot_pose_x = x
        self.robot_pose_y = y
        self.robot_pose_yaw = self.mod_yaw(yaw)

        # update the pose covariance
        G = np.array([[1.0, 0.0, -delta_dist * math.sin(self.robot_pose_yaw)],
                      [0.0, 1.0, delta_dist * math.cos(self.robot_pose_yaw)],
                      [0.0, 0.0, 1.0]])
        V = np.array([[math.cos(self.robot_pose_yaw), 0.0],
                      [math.sin(self.robot_pose_yaw), 0.0],
                      [0.0, 1.0]])
        delta_dist2 = delta_dist * delta_dist
        delta_yaw2 = delta_yaw * delta_yaw
        M = np.array([[self.odom_noise1 * delta_dist2 + self.odom_noise2 * delta_yaw2, 0.0],
                      [0.0, self.odom_noise3 * delta_dist2 + self.odom_noise4 * delta_yaw2]])
        pose_cov = np.dot(G, np.dot(self.pose_cov, G.transpose())) + np.dot(V, np.dot(M, V.transpose()))
        self.pose_cov = copy.copy(pose_cov)

    def update(self, measurements):
        trace = np.trace(self.pose_cov)
        if trace < self.min_trace:
            return
        for i in range(len(measurements)):
            if measurements[i][0] < 0.0:
                continue
            myaw = self.robot_pose_yaw + measurements[i][1]
            mx = measurements[i][0] * math.cos(myaw) + self.robot_pose_x
            my = measurements[i][0] * math.sin(myaw) + self.robot_pose_y
            min_dl = 0.0
            lidx = 0
            for j in range(len(measurements)):
                dx = self.landmarks[j][0] - mx
                dy = self.landmarks[j][1] - my
                dl = math.sqrt(dx * dx + dy * dy)
                if j == 0:
                    min_dl = dl
                elif min_dl > dl:
                    min_dl = dl
                    lidx = j
            if min_dl >= 0.5:
                continue
            dx = self.landmarks[lidx][0] - self.robot_pose_x
            dy = self.landmarks[lidx][1] - self.robot_pose_y
            q = dx * dx + dy * dy
            dyaw = math.atan2(dy, dx) - self.robot_pose_yaw
            dyaw = self.mod_yaw(dyaw)
            dz = np.array([measurements[i][0] - math.sqrt(q), measurements[i][1] - dyaw])
            dz[1] = self.mod_yaw(dz[1])
            H = np.array([[-dx / math.sqrt(q), -dy / math.sqrt(q), 0.0],
                          [dy / q, dx / q, -1.0]])
            S = np.dot(H, np.dot(self.pose_cov, H.transpose())) + self.Q
            det_S = np.linalg.det(S)
            if det_S <= 0.0:
                continue
            K = np.dot(self.pose_cov, np.dot(H.transpose(), np.linalg.inv(S)))
            mu = np.dot(K, dz)
            x = self.robot_pose_x + mu[0]
            y = self.robot_pose_y + mu[1]
            yaw = self.robot_pose_yaw + mu[2]
            yaw = self.mod_yaw(yaw)
            pose_cov = np.dot(np.eye(3) - np.dot(K, H), self.pose_cov)
            self.robot_pose_x = x
            self.robot_pose_y = y
            self.robot_pose_yaw = yaw
            self.pose_cov = pose_cov
            trace = np.trace(self.pose_cov)
            if trace < self.min_trace:
                return

    def plot_ekf_world(self, measurements):
        # clear
        plt.clf()

        # set the plot parameters
        plt.axes().set_aspect('equal')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.xlim(self.robot_pose_x - self.plot_size_x, self.robot_pose_x + self.plot_size_x)
        plt.ylim(self.robot_pose_y - self.plot_size_y, self.robot_pose_y + self.plot_size_y)
        plt.grid(which='major', color='black', linestyle='-')

        # plot the landmarks
        for i in range(len(self.landmarks)):
            plt.plot(self.landmarks[i][0], self.landmarks[i][1], marker='s', color='black', markersize=20)

        # plot the measurements according to the estimated pose
        for i in range(len(measurements)):
            myaw = self.robot_pose_yaw + measurements[i][1]
            mx = measurements[i][0] * math.cos(myaw) + self.robot_pose_x
            my = measurements[i][0] * math.sin(myaw) + self.robot_pose_y
            plt.plot(mx, my, marker='o', color='red', markersize=10)

        # plot the estimated pose
        x = self.robot_pose_x
        y = self.robot_pose_y
        dx = 0.9 * math.cos(self.robot_pose_yaw)
        dy = 0.9 * math.sin(self.robot_pose_yaw)
        plt.arrow(x=x, y=y, dx=dx, dy=dy, width=0.1, head_width=0.5, head_length=0.3, length_includes_head=True, color='green')

        # show the figure
        plt.pause(0.01)
