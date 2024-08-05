import os
import numpy as np
import matplotlib.pyplot as plt
import math

#MUDR24-304-base
#MUDR24-304-EC-2 

close_to_inner_line = 0.90
racing_line_smoothing_steps = 1 

minimum_speed = 2.1
maximum_speed = 4.0
maximum_direction_difference = 30.0

maximum_steps_to_decay_penalty = 10    # less than equal 0 disables off track panelty
maximum_steps_to_progress_ratio = 1.8   # desired max steps to be taken for 1% of progress
racing_line_free_zone_tolerance = 0.10
racing_line_safe_zone_tolerance = 0.25
distance_cover_sensitivity = 3.00  # greater value, greater agility on the track (may cause zig-zags)
speed_action_sensitivity = 3.00  # greater value increases penalty for low speed
steering_action_sensitivity = 2.00  # lower number increase penalty for not following track direction
total_off_track_penalty = 0.999999  # maximum penalty in percentage of total reward for being off track
total_bad_speed_penalty = 0.500000  # maximum penalty in percentage of total reward for being off track
total_bad_steering_penalty = 0.35  # maximum penalty in percentage of total reward for off direction steering

reward_weight_for_progrress = 35
reward_weight_for_speed = 25
reward_weight_for_steering = 20
reward_weight_for_ontrack = 15
maximum_total_reward = reward_weight_for_ontrack + reward_weight_for_progrress + reward_weight_for_steering + reward_weight_for_speed

# static
race_line = None
was_off_track_at_step = -maximum_steps_to_decay_penalty
previous_steps_reward = maximum_total_reward

# pp: Weight for the point two steps behind the current point.
# p: Weight for the point one step behind the current point.
# c: Weight for the current point.
# n: Weight for the point one step ahead of the current point.
# nn: Weight for the point two steps ahead of the current point.
def smooth_race_line(center_line, max_offset, smoothed_race_line, pp, p, c, n, nn, skip_step):
    length = len(center_line)
    new_line = [[0.0 for _ in range(2)] for _ in range(length)]
    for i in range(0, length):
        wpp = smoothed_race_line[(i - 2 * skip_step + length) % length]
        wp = smoothed_race_line[(i - skip_step + length) % length]
        wc = smoothed_race_line[i]
        wn = smoothed_race_line[(i + skip_step) % length]
        wnn = smoothed_race_line[(i + 2 * skip_step) % length]
        new_line[i][0] = pp * wpp[0] + p * wp[0] + c * wc[0] + n * wn[0] + nn * wnn[0]
        new_line[i][1] = pp * wpp[1] + p * wp[1] + c * wc[1] + n * wn[1] + nn * wnn[1]
        while calc_distance(new_line[i], center_line[i]) >= max_offset:
            new_line[i][0] = (0.98 * new_line[i][0]) + (0.02 * center_line[i][0])
            new_line[i][1] = (0.98 * new_line[i][1]) + (0.02 * center_line[i][1])
    return new_line

def calc_distance(prev_point, next_point):
    delta_x = next_point[0] - prev_point[0]
    delta_y = next_point[1] - prev_point[1]
    return math.hypot(delta_x, delta_y)

def get_race_line(center_line, max_offset, pp=0.10, p=0.05, c=0.70, n=0.05, nn=0.10, iterations=72, skip_step=1):
    if max_offset < 0.0001:
        return center_line
    if skip_step < 1:
        skip_step = 1
    smoothed_race_line = center_line
    for i in range(0, iterations):
        smoothed_race_line = smooth_race_line(center_line, max_offset, smoothed_race_line, pp, p, c, n, nn, skip_step)
    return smoothed_race_line

def calc_direction_angle(prev_point, next_point):
    return math.degrees(math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]))

def calc_direction_diff(steering, heading, track_direction):
    # Calculate the difference between the track direction and the heading direction of the car
    direction_diff = steering + heading - track_direction
    if direction_diff > 180.0:
        direction_diff = direction_diff - 360.0
    if direction_diff < -180.0:
        direction_diff = direction_diff + 360.0
    return abs(direction_diff)
    
def calc_distance_from_line(curr_point, prev_point, next_point):
    distance_cp_to_pp = calc_distance(curr_point, prev_point)  # b
    distance_cp_to_np = calc_distance(curr_point, next_point)  # a
    distance_pp_to_np = calc_distance(prev_point, next_point)  # c
    angle_pp = math.acos((distance_cp_to_pp * distance_cp_to_pp + distance_pp_to_np * distance_pp_to_np
                          - distance_cp_to_np * distance_cp_to_np) / (2 * distance_cp_to_pp * distance_pp_to_np))
    return distance_cp_to_pp * math.sin(angle_pp)

def exponential_moving_avegare(prev, new, period):
    k = 2.0 / (1.0 + period)
    return (new - prev) * k + prev


def reward_function(params):
    track_width = params['track_width']
    waypoints = params['waypoints']
    # initialize central line
    global race_line
    if race_line is None:
        max_offset = track_width * close_to_inner_line * 0.5
        race_line = get_race_line(waypoints, max_offset, skip_step=racing_line_smoothing_steps)

    global was_off_track_at_step
    steps = params['steps']
    if steps < was_off_track_at_step:
        was_off_track_at_step = -maximum_steps_to_decay_penalty
    if not params['all_wheels_on_track']:
        was_off_track_at_step = steps

    global previous_steps_reward
    if steps <= 2:
        previous_steps_reward = maximum_total_reward

    wheels_off_track_penalty = 1.0
    if maximum_steps_to_decay_penalty > 0:
        wheels_off_track_penalty = min(steps - was_off_track_at_step, maximum_steps_to_decay_penalty) / (
            1.0 * maximum_steps_to_decay_penalty)

    wp_length = len(race_line)
    wp_indices = params['closest_waypoints']
    curr_point = [params['x'], params['y']]
    prev_point = race_line[wp_indices[0]]
    next_point_1 = race_line[(wp_indices[1] + 1) % wp_length]
    next_point_2 = race_line[(wp_indices[1] + 2) % wp_length]
    next_point_3 = race_line[(wp_indices[1] + 3) % wp_length]
    track_direction_1 = calc_direction_angle(prev_point, next_point_1)
    track_direction_2 = calc_direction_angle(prev_point, next_point_2)
    track_direction_3 = calc_direction_angle(prev_point, next_point_3)

    heading = params['heading']
    steering = params['steering_angle']
    direction_diff_ratio = (
            0.20 * min((calc_direction_diff(steering, heading, track_direction_1) / maximum_direction_difference), 1.00) +
            0.30 * min((calc_direction_diff(steering, heading, track_direction_2) / maximum_direction_difference), 1.00) +
            0.50 * min((calc_direction_diff(steering, heading, track_direction_3) / maximum_direction_difference), 1.00))
    dir_steering_ratio = 1.0 - pow(direction_diff_ratio, steering_action_sensitivity)
    reward_dir_steering = reward_weight_for_steering * dir_steering_ratio

    speed = params['speed']
    expect_speed_ratio = 1.0 - min(abs(track_direction_1 - track_direction_3), maximum_direction_difference) / maximum_direction_difference
    actual_speed_ratio = max(min(speed - minimum_speed, 0), maximum_speed - minimum_speed) / (maximum_speed - minimum_speed)
    speed_ratio = 1.0 - abs(expect_speed_ratio - actual_speed_ratio)
    reward_exp_speed = reward_weight_for_speed * pow(speed_ratio, speed_action_sensitivity)

    free_zone = track_width * racing_line_free_zone_tolerance * 0.5
    safe_zone = track_width * racing_line_safe_zone_tolerance * 0.5
    dislocation = calc_distance_from_line(curr_point, prev_point, next_point_1)
    on_track_ratio = 0.0
    if dislocation <= free_zone:
        on_track_ratio = 1.0
    elif dislocation <= safe_zone:
        on_track_ratio = 1.0 - pow(dislocation / safe_zone, distance_cover_sensitivity)
    reward_on_track = on_track_ratio * reward_weight_for_ontrack

    progress = params['progress']
    reward_prog_step = reward_weight_for_progrress * min(1.0, maximum_steps_to_progress_ratio * (progress / steps))

    reward_total = reward_on_track + reward_exp_speed + reward_dir_steering + reward_prog_step
    reward_total -= reward_total * (1.0 - dir_steering_ratio) * total_bad_steering_penalty
    reward_total -= reward_total * (1.0 - on_track_ratio) * total_off_track_penalty
    reward_total -= reward_total * (1.0 - speed_ratio) * total_bad_speed_penalty
    reward_total *= wheels_off_track_penalty

    previous_steps_reward = exponential_moving_avegare(previous_steps_reward, reward_total, 3)
    return float(0.0000001 + previous_steps_reward)
