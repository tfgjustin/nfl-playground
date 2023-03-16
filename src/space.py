import math
import numpy as np

from scipy.special import expit
from scipy.stats import multivariate_normal
from tracking import is_football


# An analysis of the motion tracking data shows that 99.99% of the time, linemen are moving at or below 9 m/s
_MAX_SPEED = 9


def distance_from_point(x, y, target_x, target_y):
    return math.sqrt(((x - target_x) ** 2) + ((y - target_y) ** 2)) ** 0.5


def scale_speed(v):
    scaled = v / _MAX_SPEED
    return scaled * scaled


def control_radius(target_distance):
    # Note that these assumptions are largely for linemen and linebackers, and not for receivers or DBs.
    # We assume a player has a radius of control of about 2 yards at a minimum (e.g., a lunge) and a maximum of about
    # 6 yards when they're far away from the QB/the ball.
    return min([6, 2 + (4 * ((target_distance / 12) ** 3))])


def rotation_matrix(player_frame_row):
    # Since we've normalized that people are moving north-south instead of east-west, we're going to expand the Y-axis
    # and contract on the X-axis
    return np.array([[ player_frame_row['ySpeedFactor'], player_frame_row['xSpeedFactor']],
                     [-player_frame_row['xSpeedFactor'], player_frame_row['ySpeedFactor']]
                     ]
                    )


def scaling_matrix(player_frame_row):
    target_distance = player_frame_row['distanceFromTarget']
    speed_ratio = scale_speed(player_frame_row['s'])
    radius = control_radius(target_distance)
    x_scale = 1e-6 + (radius - (radius * speed_ratio)) / 2
    y_scale = 1e-6 + (radius + (radius * speed_ratio)) / 2
    return np.array([[x_scale, 0], [0, y_scale]])


def influence_center(player_frame_row):
    return [player_frame_row['normX'] + player_frame_row['xSpeed'] * 0.5,
            player_frame_row['normY'] + player_frame_row['ySpeed'] * 0.5
            ]


def covariance_matrix(player_frame_row):
    r = rotation_matrix(player_frame_row)
    r_inv = np.linalg.inv(r)
    s = scaling_matrix(player_frame_row)
    return r.dot(s).dot(s).dot(r_inv)


def get_target_location(frame_df):
    target_df = frame_df[frame_df.pff_positionLinedUp == 'QB']
    if target_df.shape[0] != 1:
        return None, None
    return target_df['normX'].iloc[0].astype(float), target_df['normY'].iloc[0].astype(float)


def append_target_distance(frame_df):
    target_x, target_y = get_target_location(frame_df)
    if target_x is None or target_y is None:
        return
    # Get everyone's distance from the target
    frame_df['distanceFromTarget'] = frame_df.apply(
        lambda obj: distance_from_point(obj['normX'], obj['normY'], target_x, target_y), axis=1
    )


def calculate_player_influence_map(frame_df, locations, influence):
    for _, row in frame_df.iterrows():
        if is_football(row):
            continue
        influence_multiplier = -1 if row['team'] == row['defensiveTeam'] else 1
        mu = influence_center(row)
        covariance = covariance_matrix(row)
        player_influence = multivariate_normal(mu, covariance).pdf(locations)
        influence += (player_influence * influence_multiplier)


def mask_influence_by_snap(frame_df, locations, influence):
    ball_snap_x = frame_df['ballSnapX'].iloc[0].astype(float)
    ball_snap_y = frame_df['ballSnapY'].iloc[0].astype(float)
    focus = multivariate_normal([ball_snap_x, ball_snap_y + 1], [[4, 0], [0, 3]]).pdf(locations)
    return focus * influence


def values_at_frame(frame_df, locations):
    append_target_distance(frame_df)
    influence = np.zeros(locations.shape[:2])
    calculate_player_influence_map(frame_df, locations, influence)
    influence = mask_influence_by_snap(frame_df, locations, influence)
    return (expit(influence) - 0.5) / 0.01


def analyze_frames(df, fq_frame_ids, locations):
    # Fully-qualified frame IDs are tuples of:
    # [game_id, play_id, frame_id]
    return dict(
        {fq_frame_id: values_at_frame(
            df[(df.gameId == fq_frame_id[0]) & (df.playId == fq_frame_id[1]) & (df.frameId == fq_frame_id[2])].copy(),
            locations) for fq_frame_id in fq_frame_ids}
    )


def contour_levels():
    return [-0.05, -0.02, -0.005, 0, 0.005, 0.02, 0.05]
