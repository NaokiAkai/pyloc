'''
Copyright (C) 2019 Naoki Akai.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at https://mozilla.org/MPL/2.0/.
'''

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# add the simulation module
import sys
sys.path.append("../robot_sim")

# import modules
import math
import time
from robot_sim import RobotSim
from ekf import EKF

# robot simulation parameters
start_x = 0.0
start_y = 0.0
start_yaw = 0.0 * math.pi / 180.0
max_measurement_range = 10.0
measurement_range_variance_sim = 0.2 * 0.2
measurement_angle_variance_sim = 3.0 * math.pi / 180.0 * 3.0 * math.pi / 180.0
sim_time_step = 0.1

# initialize the robot simulator
robot_sim = RobotSim(start_x, start_y, start_yaw)
robot_sim.add_landmark(2.0, 2.0)
robot_sim.add_landmark(4.0, -4.0)
robot_sim.add_landmark(-2.0, -2.0)
robot_sim.add_landmark(-4.0, 4.0)
robot_sim.set_odom_noises(0.33, 0.1, 0.1, 0.33)
robot_sim.set_max_measurement_range(max_measurement_range)
robot_sim.set_measurement_variances(measurement_range_variance_sim, measurement_angle_variance_sim)
robot_sim.set_plot_sizes(max_measurement_range, max_measurement_range)
robot_sim.set_sim_time_step(sim_time_step)

# ekf parameters
min_trace = 0.001

ekf = EKF(start_x, start_y, start_yaw)
ekf.add_landmark(2.0, 2.0)
ekf.add_landmark(4.0, -4.0)
ekf.add_landmark(-2.0, -2.0)
ekf.add_landmark(-4.0, 4.0)
ekf.set_min_trace(min_trace)
ekf.set_plot_sizes(5.0, 5.0)

while True:
    # simulate robot behaviors
    robot_sim.update_pose(0.2, -0.2)
    delta_dist = robot_sim.sim_v * robot_sim.sim_time_step
    delta_yaw = robot_sim.sim_w * robot_sim.sim_time_step
    measurements = robot_sim.get_sensor_measurements()

    # ekf
    ekf.predict(delta_dist, delta_yaw)
    ekf.update(measurements)
    ekf.print_estimated_pose()
    ekf.print_estimated_covariance()
    ekf.plot_ekf_world(measurements)

    # sleep
    time.sleep(robot_sim.sim_time_step)
