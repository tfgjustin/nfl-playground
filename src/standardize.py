import math
import numpy as np
import pandas as pd


def join_position_data_pff(pff_df, tracking_df):
    role_df = pff_df[['gameId', 'playId', 'nflId', 'pff_positionLinedUp']]
    role_df = role_df.rename(columns={'pff_positionLinedUp': 'playerPosition'}, errors='ignore')
    return pd.merge(tracking_df, role_df, on=['gameId', 'playId', 'nflId'], how='left')


def join_position_data_players(players_df, tracking_df):
    role_df = players_df[['nflId', 'PositionAbbr']]
    role_df = role_df.rename(columns={'PositionAbbr': 'playerPosition'}, errors='ignore')
    return pd.merge(tracking_df, role_df, on=['nflId'], how='left')


def join_position_data(players_df, pff_df, tracking_df):
    if pff_df is None or pff_df.empty:
        return join_position_data_players(players_df, tracking_df)
    else:
        return join_position_data_pff(pff_df, tracking_df)


def get_team_on_right(row):
    if row['team'] == 'home':
        return row['homeTeamAbbr']
    elif row['team'] == 'away':
        return row['visitorTeamAbbr']
    else:
        return np.nan


def is_reversed_special_teams_direction(row):
    if row['SpecialTeamsPlayType'] is None:
        return False
    return row['SpecialTeamsPlayType'] in ('Kickoff', 'Punt')


def infer_play_direction(row):
    if pd.isna(row['teamRightAbbr']):
        return np.nan
    elif row['possessionTeam'] == row['teamRightAbbr']:
        if is_reversed_special_teams_direction(row):
            return 'right'
        else:
            return 'left'
    elif row['possessionTeam'] != row['teamRightAbbr']:
        if is_reversed_special_teams_direction(row):
            return 'left'
        else:
            return 'right'


def swap_in_team_abbreviations(row):
    if row['team'] == 'football':
        return 'football'
    elif row['team'] == 'home':
        return row['homeTeamAbbr']
    elif row['team'] == 'away':
        return row['visitorTeamAbbr']
    else:
        return None


def maybe_insert_play_direction(games_df, plays_df, tracking_df):
    # If the tracking data doesn't have play direction, insert it.
    if 'playDirection' in tracking_df:
        return tracking_df
    # First, figure out the abbreviations of the home and visitor teams
    game_teams_df = games_df[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']]
    # At the first frame for each play, get the average X-position of each team (excluding the football)
    avg_x_by_team_df = tracking_df[(tracking_df.team != 'football') & (tracking_df['frameId'] == 1)][
        ['gameId', 'playId', 'team', 'x']
    ].groupby(['gameId', 'playId', 'team']).agg(team_x=('x', 'mean')).reset_index()
    # The avg_x_by_team_df has one row per play, with the average X coordinate of each team (excluding the ball)
    # The max_x_by_play_df dataframe has rows of (gameId, playId, max_x)
    max_x_by_play_df = avg_x_by_team_df.groupby(['gameId', 'playId']).agg(max_x=('team_x', 'max')).reset_index()
    # Now join these in to the avg_x_by_team_df to create rows of (gameId, playId, team, team_x, max_x)
    avg_max_x_by_play_df = pd.merge(avg_x_by_team_df, max_x_by_play_df, on=['gameId', 'playId'], how='left')
    # Now filter down to rows which have the team on the right.
    team_on_right_df = avg_max_x_by_play_df[avg_max_x_by_play_df.team_x == avg_max_x_by_play_df.max_x][
        ['gameId', 'playId', 'team']
    ]
    # Merge in the home and away abbreviations
    team_on_right_df = pd.merge(team_on_right_df, game_teams_df, on=['gameId'], how='left')
    # We now have the abbreviation of the team on the right
    team_on_right_df['teamRightAbbr'] = team_on_right_df.apply(get_team_on_right, axis=1)
    # Now join this with the per-play data since that does have abbreviations
    merged_df = pd.merge(plays_df, team_on_right_df, on=['gameId', 'playId'], how='left')
    # Infer the play direction based on this.
    merged_df['playDirection'] = merged_df.apply(infer_play_direction, axis=1)
    merged_df = merged_df[['gameId', 'playId', 'playDirection']]
    tracking_df = pd.merge(tracking_df, merged_df, on=['gameId', 'playId'], how='left')
    # Swap out 'home' and 'away' as teams, and replace them with the actual team abbreviations.
    # Get the tracking data per-play
    tracking_df = pd.merge(tracking_df, game_teams_df, on=['gameId'], how='left')
    # TODO: Speed this up. It's slow.
    tracking_df['teamAbbr'] = tracking_df[['team', 'homeTeamAbbr', 'visitorTeamAbbr']].apply(swap_in_team_abbreviations,
                                                                                             axis=1)
    tracking_df.drop(columns=['team', 'homeTeamAbbr', 'visitorTeamAbbr'], inplace=True)
    tracking_df = tracking_df.rename(columns={'teamAbbr': 'team'}, errors='ignore')
    return tracking_df


def _rotate_direction_offset(play_direction):
    if play_direction == 'right':
        return -90
    elif play_direction == 'left':
        return 90
    else:
        return 0


def rotate_field(tracking_df):
    # NOTE: This function REQUIRES the playDirection column, which is inserted using the games_df and plays_df data.
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


def decompose_motion_vectors(tracking_df):
    # NOTE: This depends on the field having been rotated.
    tracking_df['radiansNormDirection'] = tracking_df['normDir'].astype(float).apply(math.radians)
    # Converts angle into an x and y component
    tracking_df['xDirectionFactor'] = tracking_df['radiansNormDirection'].astype(float).apply(math.sin)
    tracking_df['yDirectionFactor'] = tracking_df['radiansNormDirection'].astype(float).apply(math.cos)
    # Determines magnitude of speed and acceleration by multiplying x and y component by magnitude of speed
    tracking_df['xSpeed'] = tracking_df['xDirectionFactor'] * tracking_df['s']
    tracking_df['ySpeed'] = tracking_df['yDirectionFactor'] * tracking_df['s']
    tracking_df['xAccel'] = tracking_df['xDirectionFactor'] * tracking_df['a']
    tracking_df['yAccel'] = tracking_df['yDirectionFactor'] * tracking_df['a']
    return tracking_df


def select_best_ball_start_xy(row):
    """
    Select the best starting coordinates for the ball for this place. Use the snap (x,y) if available, else first frame.

    :param row: One row which has both the snap (x,y) coordinates and the first (x,y) coordinates
    :return: A list of the best (x,y) coordinates.
    """
    best_x = row['ballSnapX'] if not pd.isna(row['ballSnapX']) else row['firstFrameX']
    best_y = row['ballSnapY'] if not pd.isna(row['ballSnapY']) else row['firstFrameY']
    return best_x, best_y


def append_ball_start_position(tracking_df):
    """
    Append columns with the starting position of the ball, either when snapped or the start of the play.

    :param tracking_df: DataFrame with full tracking data.
    :return: DataFrame with full tracking data, and columns showing the (x,y) coordinates of where the balls started.
    """
    # First, get all the frames representing the start of plays.
    ball_first_frames_df = tracking_df[(tracking_df.team == 'football') & (tracking_df.frameId == 1)]
    # Whittle down to just (game, play, x, y)
    ball_first_frames_df = ball_first_frames_df[['gameId', 'playId', 'normX', 'normY']]
    # Rename the coordinates to indicate this represents where the ball started.
    ball_first_frames_df.rename(columns={'normX': 'firstFrameX', 'normY': 'firstFrameY'}, inplace=True)
    # Now, get all frames where the ball was snapped.
    ball_snap_frames_df = tracking_df[(tracking_df.team == 'football') & (tracking_df.event == 'ball_snap')]
    # Whittle down to just (game, play, x, y)
    ball_snap_frames_df = ball_snap_frames_df[['gameId', 'playId', 'normX', 'normY']]
    # Rename the coordinates to indicate this represents where the ball was snapped.
    ball_snap_frames_df.rename(columns={'normX': 'ballSnapX', 'normY': 'ballSnapY'}, inplace=True)
    # Using the first frames as a base (because there's frameId==1 for every play, but not necessarily a ball snap on
    # every play) merge the ball snap into that.
    # There will be one output row for each play.
    best_ball_start_df = pd.merge(ball_first_frames_df, ball_snap_frames_df, on=['gameId', 'playId'], how='left')
    # Create a DataFrame with the (x,y) of the best ball start on each row.
    best_ball_start_xy_df = best_ball_start_df.apply(
        lambda row: select_best_ball_start_xy(row), result_type='expand', axis=1
    )
    # Merge in the best position using the index values (i.e., ordered play number) so we now have one row per
    # (game, play) that has the best starting (x, y) coordinates for the ball.
    best_ball_start_df = best_ball_start_df.merge(best_ball_start_xy_df, how='left', left_index=True, right_index=True)
    # Rename the columns to be descriptive.
    best_ball_start_df = best_ball_start_df.rename(columns={0: 'ballStartX', 1: 'ballStartY'})
    # Drop the irrelevant columns.
    best_ball_start_df.drop(columns=['firstFrameX', 'firstFrameY', 'ballSnapX', 'ballSnapY'], inplace=True)
    # Now merge in the best ball start position, so each row in the tracking data has where the ball started that play.
    return pd.merge(tracking_df, best_ball_start_df, on=['gameId', 'playId'], how='left')


def annotate_tracking_data(tracking_df):
    tracking_df = rotate_field(tracking_df)
    tracking_df = decompose_motion_vectors(tracking_df)
    tracking_df = append_ball_start_position(tracking_df)
    return tracking_df


def standardize_tracking_dataframes(games_df, plays_df, tracking_df, pff_df=None, players_df=None):
    tracking_df = join_position_data(players_df, pff_df, tracking_df)
    tracking_df = maybe_insert_play_direction(games_df, plays_df, tracking_df)
    tracking_df = annotate_tracking_data(tracking_df)
    return tracking_df


def try_read_pff(filename):
    try:
        return pd.read_csv(filename)
    except pd.errors.EmptyDataError:
        return None
