#!/usr/bin/python

import erylib
import pandas as pd
import numpy as np
import sys

from common import setup_environment
from pass_accumulator import PassAccumulator
from plays import load_plays_data
from standardize import normalize_column_formatting, standardize_tracking_dataframes, try_read_pff
from keras.models import load_model
from time import localtime, strftime, time
from tracking import load_all_tracking_data

_POSITIONS_OFFENSE = ['RB', 'TE', 'WR']
_POSITIONS_DEFENSE = ['DB', 'LB']


def absolute_score_margin(row):
    return abs(row['preSnapHomeScore'] - row['preSnapVisitorScore'])


def offense_score_margin(row):
    margin = row['preSnapHomeScore'] - row['preSnapVisitorScore']
    return margin if row['possessionTeam'] == row['homeTeamAbbr'] else -margin


def convert_clock(clock_face):
    mins, secs = clock_face.split(':', maxsplit=1)
    return 60 * int(mins) + int(secs)


def seconds_left_in_half(row):
    quarter = row['quarter']
    clock_face = row['gameClock']
    half_quarter = 1 - ((quarter - 1) % 2)
    return 900 * half_quarter + convert_clock(clock_face)


def filter_plays(df):
    # We want plays:
    # - During regulation
    df = df[df.quarter <= 4]
    # - Between the 20 yard lines (i.e. within 30 yards of the middle of the field
    df = df[abs(df.absoluteYardlineNumber - 50) <= 30]
    df['offenseScoreMargin'] = df.apply(lambda row: offense_score_margin(row), axis=1)
    # - Score differential within 3 possessions (24 points)
    df['absoluteScoreMargin'] = df.apply(lambda row: absolute_score_margin(row), axis=1)
    df = df[df.absoluteScoreMargin <= 24]
    # - Outside the 2-minute warning
    df['secondsLeftInHalf'] = df.apply(lambda row: seconds_left_in_half(row), axis=1)
    df = df[df.secondsLeftInHalf >= 120]
    return df


def get_position_count(lineup, positions):
    counts = {p: 0 for p in positions}
    if pd.isna(lineup):
        return [0] * len(positions)
    for val in lineup.split(', '):
        count, position = val.split(maxsplit=1)
        counts[position] = int(count)
    return [counts[p] for p in sorted(positions)]


def position_column_name(side, position):
    return '%sNum%s' % (side, position)


def position_column_names(side, positions):
    return [position_column_name(side, position) for position in sorted(positions)]


def decompose_positions(df, input_column_suffix, side, positions):
    input_column_name = 'personnel' + input_column_suffix
    df.loc[:, position_column_names(side, positions)] = [get_position_count(p, positions)
                                                         for p in df[input_column_name]]
    return df


def order_plays(df):
    df.sort_values(by=['gameId', 'quarter', 'secondsLeftInHalf'], ascending=[True, True, False], inplace=True,
                   na_position='last')


def extract_personnel_from_plays(df):
    df = decompose_positions(df, 'O', 'offense', _POSITIONS_OFFENSE)
    df = decompose_positions(df, 'D', 'defense', _POSITIONS_DEFENSE)
    return df


def get_post_snap_frames(merged_df):
    # TODO: Create a DataFrame which is just a tuple of (gameId, playId, frameId) we want to keep, join, and return it.
    ball_snap_frames = merged_df[(merged_df.team == 'football') & (merged_df.event == 'ball_snap')]
    # Make sure we make a copy of the input DataFrame so we don't accidentally update the original frame counts.
    ball_snap_frames = ball_snap_frames[['gameId', 'playId', 'frameId']].copy()
    # Target 0.8s after the ball is snapped.
    ball_snap_frames['targetFrameId'] = ball_snap_frames['frameId'] + 8
    return ball_snap_frames.drop(columns='frameId')


def get_pass_accumulator_df(games_df, plays_df):
    annotator = PassAccumulator(games_df, plays_df)
    annotator.annotate()
    return annotator.df()


def select_and_decompose_plays(games_df, plays_df):
    early_games_df = games_df[games_df.week <= 6]
    print('Found %d games' % len(early_games_df))
    early_game_plays_df = pd.merge(early_games_df, plays_df, on=['gameId'], how='left')
    print('Found %d plays' % len(early_game_plays_df))
    filtered_plays_df = filter_plays(early_game_plays_df)
    print('Filtered down to %d plays' % len(filtered_plays_df))
    order_plays(filtered_plays_df)
    return extract_personnel_from_plays(filtered_plays_df)


def annotate_game_plays_with_tracking(game_plays_df, pff_df, players_df, tracking_df):
    merged_df = pd.merge(game_plays_df, tracking_df, on=['gameId', 'playId'], how='left')
    print('Plays have %d tracking items' % len(merged_df))
    post_snap_frames = get_post_snap_frames(merged_df)
    merged_df = pd.merge(merged_df, post_snap_frames, on=['gameId', 'playId'], how='left')
    merged_df = merged_df[abs(merged_df.frameId - merged_df.targetFrameId) <= 2]
    print('Filtered down to %d tracking items' % len(merged_df))
    # TODO: Find a way to merge in fewer columns
    merged_df = pd.merge(merged_df, players_df, on=['nflId'], how='left')
    if pff_df is not None:
        merged_df = pd.merge(merged_df, pff_df, on=['gameId', 'playId', 'nflId'], how='left')
    merged_df['isBallCarrier'] = merged_df.positionAbbr == 'QB'
    merged_df[merged_df.isBallCarrier, 's'] = 0
    merged_df['isOffense'] = merged_df.possessionTeam == merged_df.team
    return merged_df


def analyze_all_plays(ery_model, merged_df):
    keys = list()
    to_predict = list()
    for name, group in merged_df.groupby(by=['gameId', 'playId', 'frameId']):
        sample = erylib.construct_samples_from_play(None, name, group, create_mirror_samples=False)
        if sample is not None:
            to_predict.extend(sample)
            keys.extend([name[0], name[1]] * len(sample))
    X, _, _ = erylib.split_x_y(to_predict)
    y_pred = ery_model.predict(X)
    yards = np.arange(-29, 51)
    summary = np.array(keys).reshape(-1, 2)
    return np.hstack([summary, np.sum(yards * y_pred, axis=1).reshape(-1, 1)])


def merge_analysis_frames(ery_df, pass_data_df, game_plays_df):
    slim_columns = ['gameId', 'playId', 'quarter', 'secondsLeftInHalf', 'offenseScoreMargin', 'down', 'yardsToGo',
                    'absoluteYardlineNumber']
    slim_columns += [position_column_name('offense', p) for p in _POSITIONS_OFFENSE]
    slim_columns += [position_column_name('defense', p) for p in _POSITIONS_DEFENSE]
    slim_merged_df = game_plays_df[slim_columns].drop_duplicates()
    analysis_df = pd.merge(ery_df, slim_merged_df, on=['gameId', 'playId'], how='inner')
    return pd.merge(analysis_df, pass_data_df, on=['gameId', 'playId'], how='inner')


def analyze_all_tracking_data(tracking_filenames, ery_model, games_df, plays_df, game_plays_df, pff_df, players_df):
    ery_array = list()
    for tracking_filename in sorted(tracking_filenames):
        print(tracking_filename)
        tracking_df = load_all_tracking_data([tracking_filename])
        tracking_df = standardize_tracking_dataframes(games_df, plays_df, tracking_df, pff_df=pff_df,
                                                      players_df=players_df)
        merged_df = annotate_game_plays_with_tracking(game_plays_df, pff_df, players_df, tracking_df)
        ery_array.extend(analyze_all_plays(ery_model, merged_df))
    ery_df = pd.DataFrame(data=ery_array, columns=['gameId', 'playId', 'ery'])
    ery_df.astype({'gameId': int, 'playId': int})
    return ery_df


def filetime_to_path(filetime):
    filetime_ns = filetime - int(filetime)
    time_struct = localtime(filetime)
    time_fmt = strftime('%Y%m%dT%H%M%S', time_struct)
    return 'input_df.%s.%d.csv' % (time_fmt, 1000000 * filetime_ns)


def create_model_input(analysis_df, filetime):
    analysis_df.to_csv(filetime_to_path(filetime), index=False)


def main(argv):
    setup_environment()
    if len(argv) < 7:
        print('Usage: %s <model_dir> <games_csv> <plays_csv> <players_csv> <pff_csv> <tracking_csv>' %
              argv[0])
        return 1
    ery_model = load_model(argv[1], compile=False, custom_objects={
        'MyLearningRateSchedule': erylib.MyLearningRateSchedule, 'CrpsLoss': erylib.CrpsLoss
    })
    games_df = normalize_column_formatting(pd.read_csv(argv[2]))
    plays_df = load_plays_data(games_df, argv[3])
    players_df = normalize_column_formatting(pd.read_csv(argv[4]))
    pff_df = try_read_pff(argv[5])
    pass_data_df = get_pass_accumulator_df(games_df, plays_df)
    game_plays_df = select_and_decompose_plays(games_df, plays_df)
    ery_df = analyze_all_tracking_data(argv[6:], ery_model, games_df, plays_df, game_plays_df, pff_df, players_df)
    analysis_df = merge_analysis_frames(ery_df, pass_data_df, game_plays_df)
    create_model_input(analysis_df, time())
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
