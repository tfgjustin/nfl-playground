import math
import numpy as np

from scipy.special import expit
from scipy.stats import multivariate_normal
from tracking import is_football


# An analysis of the motion tracking data shows that 99.99% of the time, linemen are moving at or below 9 m/s
_MAX_SPEED = 9


def create_uniform_mask(locations):
    mask_shape = locations.shape[:2]
    uniform_mask = np.ndarray(mask_shape, dtype=np.float)
    mask_value = 1. / uniform_mask.size
    uniform_mask.fill(mask_value)
    return uniform_mask


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
    return np.array([[player_frame_row['yDirectionFactor'], player_frame_row['xDirectionFactor']],
                     [-player_frame_row['xDirectionFactor'], player_frame_row['yDirectionFactor']]
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
    # The center of influence is (heuristically) where the player will be in 0.5 seconds.
    # Since some data also includes acceleration information, we can also use that to calculate displacement.
    # x = v*t + 0.5*a*(t^2)
    #   = v*(0.5) + 0.5*a*(0.5^2)
    #   = v*0.5 + 0.125*a
    return [player_frame_row['normX'] + player_frame_row['xSpeed'] * 0.5 + player_frame_row['xAccel'] * 0.125,
            player_frame_row['normY'] + player_frame_row['ySpeed'] * 0.5 + player_frame_row['yAccel'] * 0.125
            ]


def covariance_matrix(player_frame_row):
    r = rotation_matrix(player_frame_row)
    r_inv = np.linalg.inv(r)
    s = scaling_matrix(player_frame_row)
    return r.dot(s).dot(s).dot(r_inv)


def get_target_location(frame_df):
    target_df = frame_df[frame_df.playerPosition == 'QB']
    if target_df.shape[0] != 1:
        target_df = frame_df[frame_df.team == 'football']
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
        player_influence = np.zeros(locations.shape[:2])
        mu = influence_center(row)
        if np.isnan(np.sum(mu)):
            print('Influence center has NaN:', mu, "; returning zeros")
            print('Row:', row)
            return player_influence
        covariance = covariance_matrix(row)
        if np.isnan(np.sum(covariance)):
            print('Covariance has NaN:', covariance, "; returning zeros")
            print('Row:', row)
            return player_influence
        try:
            player_influence = multivariate_normal(mu, covariance).pdf(locations)
        except np.linalg.LinAlgError:
            print('Covariance matrix for player influence is singular; returning zeros')
            print('Cov:', covariance)
            print('Row:', row)
        if np.isnan(np.sum(player_influence)):
            print('NaN in player influence map; returning zeros')
            print('== Mu:', mu, ' Cov:', covariance)
            print('Row:', row)
            player_influence = np.zeros(locations.shape[:2])
        influence += (player_influence * influence_multiplier)


def should_mask_influence_by_snap(_):
    return True


def mask_influence_uniform(influence, uniform_mask):
    return uniform_mask * influence


def mask_influence_by_snap(fq_frame_id, frame_df, locations, influence):
    ball_snap_x = frame_df['ballStartX'].iloc[0].astype(float)
    ball_snap_y = frame_df['ballStartY'].iloc[0].astype(float)
    if np.isnan(ball_snap_x) or np.isnan(ball_snap_y):
        print('NaN for ball snap in frame', fq_frame_id)
        return influence
    focus = multivariate_normal([ball_snap_x, ball_snap_y], [[4, 0], [0, 2]]).pdf(locations)
    return focus * influence


def mask_influence(fq_frame_id, frame_df, locations, influence, uniform_mask):
    if should_mask_influence_by_snap(frame_df):
        return mask_influence_by_snap(fq_frame_id, frame_df, locations, influence)
    else:
        return mask_influence_uniform(influence, uniform_mask)


def values_at_frame(fq_frame_id, frame_df, locations, uniform_mask):
    append_target_distance(frame_df)
    influence = np.zeros(locations.shape[:2])
    calculate_player_influence_map(frame_df, locations, influence)
    influence = mask_influence(fq_frame_id, frame_df, locations, influence, uniform_mask)
    return (expit(influence) - 0.5) / 0.01


def analyze_frames(df, fq_frame_ids, locations):
    # Fully-qualified frame IDs are tuples of:
    # [game_id, play_id, frame_id]
    uniform_mask = create_uniform_mask(locations)
    return dict(
        {fq_frame_id: values_at_frame(
            fq_frame_id,
            df[(df.gameId == fq_frame_id[0]) & (df.playId == fq_frame_id[1]) & (df.frameId == fq_frame_id[2])].copy(),
            locations, uniform_mask) for fq_frame_id in fq_frame_ids}
    )


def contour_levels():
    # return [-0.0005, -0.0002, -0.00005, 0, 0.00005, 0.0002, 0.0005]
    return [-0.05, -0.02, -0.005, 0, 0.005, 0.02, 0.05]
    # return [-5, -2, -0.5, 0, 0.5, 2, 5]
