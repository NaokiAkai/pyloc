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
from mcl import MCL

# robot simulation parameters
start_x = 0.0
start_y = 0.0
start_yaw = 0.0 * math.pi / 180.0
max_measurement_range = 10.0
measurement_range_variance_sim = 0.2 * 0.2
measurement_angle_variance_sim = 3.0 * math.pi / 180.0 * 3.0 * math.pi / 180.0
random_measurement_rate = 0.1
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
robot_sim.set_random_measurement_rate(random_measurement_rate)
robot_sim.set_plot_sizes(max_measurement_range, max_measurement_range)
robot_sim.set_sim_time_step(sim_time_step)

# mcl parameters
particle_num = 50
measurement_variance = 0.3 * 0.3
measurement_resolution = 0.1
z_hit = 0.9
z_rand = 0.1
initial_var_x = 0.2
initial_var_y = 0.2
initial_var_yaw = 2.0 * math.pi / 180.0

# initialize the mcl
mcl = MCL(start_x, start_y, start_yaw)
mcl.set_particle_num(particle_num)
mcl.set_odom_noises(20.0 , 10.0, 10.0, 30.0)
mcl.add_landmark(2.0, 2.0)
mcl.add_landmark(4.0, -4.0)
mcl.add_landmark(-2.0, -2.0)
mcl.add_landmark(-4.0, 4.0)
mcl.set_measurement_variance(measurement_variance)
mcl.set_measurement_resolution(measurement_resolution)
mcl.set_measurement_model_coefficients(z_hit, z_rand)
mcl.set_resample_threshold(0.5)
mcl.set_plot_sizes(5.0, 5.0)
mcl.reset_particles(initial_var_x, initial_var_y, initial_var_yaw)

while True:
    # simulate robot behaviors
    robot_sim.update_pose(0.2, -0.2)
    delta_dist = robot_sim.sim_v * robot_sim.sim_time_step
    delta_yaw = robot_sim.sim_w * robot_sim.sim_time_step
    measurements = robot_sim.get_sensor_measurements()

    # mcl
    mcl.update_particles(delta_dist, delta_yaw)
    mcl.calculate_weights(measurements)
    mcl.estimate_robot_pose()
    mcl.resample_particles()
    mcl.print_estimated_pose()
    mcl.print_effective_sample_size_and_total_weight()
    mcl.plot_mcl_world(measurements)

    # sleep
    time.sleep(robot_sim.sim_time_step)
