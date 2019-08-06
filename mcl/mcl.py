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
import matplotlib.pyplot as plt

class MCL:
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.robot_pose_x = x
        self.robot_pose_y = y
        self.robot_pose_yaw = yaw

        self.odom_noise1 = 1.0
        self.odom_noise2 = 0.5
        self.odom_noise3 = 0.5
        self.odom_noise4 = 2.0
        self.measurement_variance = 0.1 * 0.1
        self.measurement_resolution = 0.1
        self.z_hit = 0.9
        self.z_rand = 0.1
        self.resample_threshold = 0.5

        self.particle_num = 100
        self.particles = []
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

    def set_particle_num(self, particle_num):
        self.particle_num = particle_num

    def set_odom_noises(self, odom_noise1, odom_noise2, odom_noise3, odom_noise4):
        self.odom_noise1 = odom_noise1
        self.odom_noise2 = odom_noise2
        self.odom_noise3 = odom_noise3
        self.odom_noise4 = odom_noise4

    def set_measurement_variance(self, measurement_variance):
        self.measurement_variance = measurement_variance

    def set_measurement_resolution(self, measurement_resolution):
        self.measurement_resolution = measurement_resolution

    def set_measurement_model_coefficients(self, z_hit, z_rand):
        self.z_hit = z_hit
        self.z_rand = z_rand
        if self.z_hit + self.z_rand != 1.0:
            print 'sum of z_hit and z_rand must be one.'
            sys.exit(-1)

    def set_resample_threshold(self, resample_threshold):
        self.resample_threshold = resample_threshold

    def set_plot_sizes(self, plot_size_x, plot_size_y):
        self.plot_size_x = plot_size_x
        self.plot_size_y = plot_size_y

    def add_landmark(self, x, y):
        self.landmarks.append([x, y])

    def print_estimated_pose(self):
        print 'x =', self.robot_pose_x, '[m], y =', self.robot_pose_y, '[m], yaw =', self.robot_pose_yaw * 180.0 / self.PI, '[deg]'

    def print_effective_sample_size_and_total_weight(self):
        print 'particle num', len(self.particles), ', effective sample size =', self.effective_sample_size, ', total weight =', self.total_weight

    def reset_particles(self, var_x, var_y, var_yaw):
        # self.particles.clear() # Python 3.3
        del self.particles[:]
        wo = 1.0 / float(self.particle_num)
        for i in range(self.particle_num):
            x = normal(self.robot_pose_x, var_x)
            y = normal(self.robot_pose_y, var_y)
            yaw = normal(self.robot_pose_yaw, var_yaw)
            yaw = self.mod_yaw(yaw)
            self.particles.append([x, y, yaw, wo])

    def update_particles(self, delta_dist, delta_yaw):
        delta_dist2 = delta_dist * delta_dist
        delta_yaw2 = delta_yaw * delta_yaw
        for i in range(self.particle_num):
            del_dist = normal(delta_dist, self.odom_noise1 * delta_dist2 + self.odom_noise2 * delta_yaw2)
            del_yaw = normal(delta_yaw, self.odom_noise3 * delta_dist2 + self.odom_noise4 * delta_yaw2)
            x = self.particles[i][0] + del_dist * math.cos(self.particles[i][2])
            y = self.particles[i][1] + del_dist * math.sin(self.particles[i][2])
            yaw_ = self.particles[i][2] + del_yaw
            yaw = self.mod_yaw(yaw_)
            self.particles[i][0] = x
            self.particles[i][1] = y
            self.particles[i][2] = yaw

    def calculate_weights(self, measurements):
        self.total_weight = 0.0
        norm_coef = 1.0 / (math.sqrt(2.0 * self.PI * self.measurement_variance))
        for i in range(self.particle_num):
            total_log_prob = 0.0
            for j in range(len(measurements)):
                myaw = self.particles[i][2] + measurements[j][1]
                mx = measurements[j][0] * math.cos(myaw) + self.particles[i][0]
                my = measurements[j][0] * math.sin(myaw) + self.particles[i][1]
                min_dl = 0.0
                for k in range(len(self.landmarks)):
                    dx = self.landmarks[k][0] - mx
                    dy = self.landmarks[k][1] - my
                    dl = math.sqrt(dx * dx + dy * dy)
                    if k == 0:
                        min_dl = dl
                    elif min_dl > dl:
                        min_dl = dl
                prob = self.z_hit * norm_coef * math.exp(-0.5 * (min_dl * min_dl) / (2.0 * self.measurement_variance)) + self.z_rand * 10e-6
                prob *= self.measurement_resolution
                if prob > 1.0:
                    prob = 1.0
                total_log_prob += math.log(prob)
            prob = math.exp(total_log_prob)
            weight = self.particles[i][3] * prob
            self.particles[i][3] = weight
            self.total_weight += weight

        # normalize weights and calculate effective sample size
        self.effective_sample_size = 0.0
        for i in range(self.particle_num):
            normalized_weight = self.particles[i][3] / self.total_weight
            self.particles[i][3] = normalized_weight
            normalized_weight2 = normalized_weight * normalized_weight
            self.effective_sample_size += normalized_weight2
        self.effective_sample_size = 1.0 / self.effective_sample_size

    def estimate_robot_pose(self):
        x = 0.0
        y = 0.0
        yaw = 0.0
        tmp_yaw = copy.copy(self.robot_pose_yaw)
        for i in range(self.particle_num):
            x += self.particles[i][0] * self.particles[i][3]
            y += self.particles[i][1] * self.particles[i][3]
            dyaw = tmp_yaw - self.particles[i][2]
            dyaw = self.mod_yaw(dyaw)
            yaw += dyaw * self.particles[i][3]
        yaw_ = tmp_yaw - yaw
        self.robot_pose_x = x
        self.robot_pose_y = y
        self.robot_pose_yaw = self.mod_yaw(yaw_)

    def resample_particles(self):
        if self.effective_sample_size > float(self.particle_num) * self.resample_threshold:
            return
        tmp_particles = copy.copy(self.particles)
        wo = 1.0 / float(self.particle_num)
        board = []
        board.append(self.particles[0][3])
        for i in range(1, self.particle_num):
            board.append(board[i - 1] + self.particles[i][3])
        for i in range(self.particle_num):
            darts = random()
            for j in range(self.particle_num):
                if (darts < board[j]):
                    self.particles[i] = copy.copy(tmp_particles[j])
                    self.particles[i][3] = wo
                    break

    def plot_mcl_world(self, measurements):
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

        # plot the particles
        for i in range(self.particle_num):
            x = self.particles[i][0]
            y = self.particles[i][1]
            dx = 0.5 * math.cos(self.particles[i][2])
            dy = 0.5 * math.sin(self.particles[i][2])
            plt.arrow(x=x, y=y, dx=dx, dy=dy, width=0.05, head_width=0.25, head_length=0.15, length_includes_head=True, color='blue')

        # show the figure
        plt.pause(0.01)
