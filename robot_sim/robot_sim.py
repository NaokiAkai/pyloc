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
from numpy.random import normal, random
import matplotlib.pyplot as plt

class RobotSim:
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.gt_x = x
        self.gt_y = y
        self.gt_yaw = yaw
        self.sim_x = x
        self.sim_y = y
        self.sim_yaw = yaw
        self.sim_v = 0.0
        self.sim_w = 0.0

        self.odom_noise1 = 1.0
        self.odom_noise2 = 0.5
        self.odom_noise3 = 0.5
        self.odom_noise4 = 2.0
        self.sim_time_step = 0.05
        self.max_measurement_range = 5.0
        self.measurement_range_variance = 0.1 * 0.1
        self.measurement_angle_variance = 0.01 * 0.01
        self.random_measurement_rate = 0.05
        self.landmarks = []

        self.plot_size_x = 5.0
        self.plot_size_y = 5.0

        self.PI = 3.14159265359
        self.PI2 = 6.28318530718

    def set_odom_noises(self, odom_noise1, odom_noise2, odom_noise3, odom_noise4):
        self.odom_noise1 = odom_noise1
        self.odom_noise2 = odom_noise2
        self.odom_noise3 = odom_noise3
        self.odom_noise4 = odom_noise4

    def set_plot_sizes(self, plot_size_x, plot_size_y):
        self.plot_size_x = plot_size_x
        self.plot_size_y = plot_size_y

    def set_sim_time_step(self, sim_time_step):
        self.sim_time_step = sim_time_step

    def set_max_measurement_range(self, max_measurement_range):
        self.max_measurement_range = max_measurement_range

    def set_measurement_variances(self, measurement_range_variance, measurement_angle_variance):
        self.measurement_range_variance = measurement_range_variance
        self.measurement_angle_variance = measurement_angle_variance

    def set_random_measurement_rate(self, random_measurement_rate):
        self.random_measurement_rate = random_measurement_rate

    def add_landmark(self, x, y):
        self.landmarks.append([x, y])

    def print_landmarks(self):
        for i in range(len(self.landmarks)):
            print i, 'th landmark: x =', self.landmarks[i][0], '[m], y =', self.landmarks[i][1], ' [m]'
        print ''

    def mod_yaw(self, yaw):
        while yaw < -self.PI:
            yaw += self.PI2
        while yaw > self.PI:
            yaw -= self.PI2
        return yaw

    def print_gt_pose(self):
        print 'gt: x =', self.gt_x, '[m], y =', self.gt_y, '[m], yaw =', self.gt_yaw * 180.0 / self.PI, '[deg]'

    def get_gt_pose(self):
        return self.gt_x, self.gt_y, self.gt_yaw

    def print_sim_pose(self):
        print 'sim: x =', self.sim_x, '[m], y =', self.sim_y, '[m], yaw =', self.sim_yaw * 180.0 / self.PI, '[deg]'

    def get_sim_pose(self):
        return self.sim_x, self.sim_y, self.sim_yaw

    def update_pose(self, v, w):
        # update the ground truth pose
        delta_dist = v * self.sim_time_step
        delta_yaw = w * self.sim_time_step
        x = self.gt_x + delta_dist * math.cos(self.gt_yaw)
        y = self.gt_y + delta_dist * math.sin(self.gt_yaw)
        yaw = self.gt_yaw + delta_yaw
        self.gt_x = x
        self.gt_y = y
        self.gt_yaw = self.mod_yaw(yaw)

        # update the simulation pose and calculate the simulation velocities
        delta_dist2 = delta_dist * delta_dist
        delta_yaw2 = delta_yaw * delta_yaw
        delta_dist_sim = delta_dist * 0.9 + normal(0.0, self.odom_noise1 * delta_dist2 + self.odom_noise2 * delta_yaw2)
        delta_yaw_sim = delta_yaw * 0.9 + normal(0.0, self.odom_noise3 * delta_dist2 + self.odom_noise4 * delta_yaw2)
        x = self.sim_x + delta_dist_sim * math.cos(self.sim_yaw)
        y = self.sim_y + delta_dist_sim * math.sin(self.sim_yaw)
        yaw = self.sim_yaw + delta_yaw_sim
        self.sim_x = x
        self.sim_y = y
        self.sim_yaw = self.mod_yaw(yaw)
        self.sim_v = delta_dist_sim / self.sim_time_step
        self.sim_w = delta_yaw_sim / self.sim_time_step

    def print_simulated_velocities(self):
        print 'v =', self.sim_v, '[m/sec], w =', self.sim_w, '[rad/sec]'

    def get_simulated_velocities(self):
        return self.sim_v, self.sim_w

    def get_sensor_measurements(self):
        measurements = []
        for i in range(len(self.landmarks)):
            dx = self.landmarks[i][0] - self.gt_x
            dy = self.landmarks[i][1] - self.gt_y
            dl = normal(math.sqrt(dx * dx + dy * dy), self.measurement_range_variance)
            if dl <= self.max_measurement_range:
                dyaw = normal(math.atan2(dy, dx) - self.gt_yaw, self.measurement_angle_variance)
                dyaw = self.mod_yaw(dyaw)
                # simulate random range measurement
                if random() < self.random_measurement_rate:
                    dl = random() * self.max_measurement_range
                measurements.append([dl, dyaw])
        return measurements

    def plot_sim_world(self, x, y, yaw, measurements):
        plt.clf()
        plt.xlim(x - self.plot_size_x, x + self.plot_size_x)
        plt.ylim(y - self.plot_size_y, y + self.plot_size_y)
        plt.grid(which='major', color='black', linestyle='-')
        # plt.grid(which='minor', color='black', linestyle='-')
        for i in range(len(self.landmarks)):
            plt.plot(self.landmarks[i][0], self.landmarks[i][1], marker='o', color='black', markersize=30)
        for i in range(len(measurements)):
            if measurements[i][0] > 0.0:
                mx = measurements[i][0] * math.cos(yaw + measurements[i][1]) + x
                my = measurements[i][0] * math.sin(yaw + measurements[i][1]) + y
                plt.plot(mx, my, marker='o', color='red', markersize=20)
        plt.plot(self.gt_x, self.gt_y, marker='o', color='black', markersize=30)
        plt.plot(x, y, marker='o', color='green', markersize=20)
        plt.pause(self.sim_time_step)
