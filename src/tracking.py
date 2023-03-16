import math
import pandas as pd


def _rotate_direction_offset(play_direction):
    if play_direction == 'right':
        return -90
    elif play_direction == 'left':
        return 90
    else:
        return 0


def is_football(row):
    return row['team'] == 'football'


def rotate_field(tracking_df):
    # Do a naive swap. This has the effect of doing a diagonal mirroring of the field.
    tracking_df['normX'] = tracking_df['y']
    tracking_df['normY'] = tracking_df['x']
    # If the play is moving to the right, then rotating everything -90 degrees means we need to mirror L-to-R to get
    # everything to look correct, by essentially "undoing" the original diagonal mirroring.
    tracking_df.loc[tracking_df.playDirection == 'right', ['normX']] = 53.5 - tracking_df['normX']
    # If the play is moving to the left, then rotating everything 90 degrees means we need to mirror top-to-bottom to
    # get everything to look correct.
    tracking_df.loc[tracking_df.playDirection == 'left', ['normY']] = 120 - tracking_df['normY']
    tracking_df['normDirOffset'] = tracking_df['playDirection'].apply(_rotate_direction_offset)
    tracking_df['normDir'] = tracking_df['dir'] + tracking_df['normDirOffset']
    return tracking_df


def decompose_speed_vectors(tracking_df):
    tracking_df['radiansNormDirection'] = tracking_df['normDir'].astype(float).apply(math.radians)
    # Converts angle into an x and y component
    tracking_df['xSpeedFactor'] = tracking_df['radiansNormDirection'].astype(float).apply(math.sin)
    tracking_df['ySpeedFactor'] = tracking_df['radiansNormDirection'].astype(float).apply(math.cos)
    # Determines magnitude of speed by multiplying x and y component by magnitude of speed
    tracking_df['xSpeed'] = tracking_df['xSpeedFactor'] * tracking_df['s']
    tracking_df['ySpeed'] = tracking_df['ySpeedFactor'] * tracking_df['s']
    return tracking_df


def append_ball_snap_position(tracking_df):
    # Make sure we make a copy of the input DataFrame so we don't accidentally update the original frame counts.
    ball_snap_frames = tracking_df[(tracking_df.team == 'football') & (tracking_df.event == 'ball_snap')].copy()
    ball_snap_frames = ball_snap_frames[['gameId', 'playId', 'normX', 'normY']]
    _ = ball_snap_frames.rename(columns={'normX': 'ballSnapX', 'normY': 'ballSnapY'}, inplace=True)
    return pd.merge(tracking_df, ball_snap_frames, on=['gameId', 'playId'], how='left')
