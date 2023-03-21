#!/usr/bin/python

import pandas as pd
import numpy as np
import sys

from animate import create_one_frame
from plays import load_plays_data
from space import analyze_frames
from standardize import standardize_all_dataframes
from tracking import load_all_tracking_data


def absolute_score_margin(row):
    return abs(row['preSnapHomeScore'] - row['preSnapVisitorScore'])


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
    df = df[abs(df.absoluteYardlineNumber - 60) <= 2]
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
    _POSITIONS_OFFENSE = ['RB', 'TE', 'WR']
    _POSITIONS_DEFENSE = ['DB', 'LB']
    df = decompose_positions(df, 'O', 'offense', _POSITIONS_OFFENSE)
    df = decompose_positions(df, 'D', 'defense', _POSITIONS_DEFENSE)
    return df


def get_post_snap_frames(merged_df):
    # Make sure we make a copy of the input DataFrame so we don't accidentally update the original frame counts.
    ball_snap_frames = merged_df[(merged_df.team == 'football') & (merged_df.event == 'ball_snap')].copy()
    ball_snap_frames = ball_snap_frames[['gameId', 'playId', 'frameId']]
    ball_snap_frames['targetFrameId'] = ball_snap_frames['frameId'] + 10
    return ball_snap_frames.drop(columns='frameId')


def select_and_merge_data(games_df, plays_df, pff_df, players_df, tracking_df):
    merged_df = games_df[games_df.week <= 6]
    print('Found %d games' % len(merged_df))
    merged_df = pd.merge(merged_df, plays_df, on=['gameId'], how='left')
    print('Found %d plays' % len(merged_df))
    # TODO: Prior to this, annotate with successful pass counts
    merged_df = filter_plays(merged_df)
    print('Filtered down to %d plays' % len(merged_df))
    order_plays(merged_df)
    merged_df = extract_personnel_from_plays(merged_df)
    merged_df = pd.merge(merged_df, tracking_df, on=['gameId', 'playId'], how='left')
    print('Plays have %d tracking items' % len(merged_df))
    post_snap_frames = get_post_snap_frames(merged_df)
    merged_df = pd.merge(merged_df, post_snap_frames, on=['gameId', 'playId'], how='left')
    merged_df = merged_df[merged_df.frameId == merged_df.targetFrameId]
    print('Filtered down to %d tracking items' % len(merged_df))
    merged_df = pd.merge(merged_df, players_df, on=['nflId'], how='left')
    if pff_df is not None:
        merged_df = pd.merge(merged_df, pff_df, on=['gameId', 'playId', 'nflId'], how='left')
    return merged_df


def summarize_frame(fq_frame_id, influence):
    min_i = np.min(influence)
    max_i = np.max(influence)
    sum_i = float(np.sum(influence, axis=None))
    print(fq_frame_id, 'Total: %6.2f [%5.3f - %5.3f]' % (sum_i, min_i, max_i))


def analyze_all_plays(merged_df):
    groups = merged_df.groupby(by=['gameId', 'playId', 'frameId'])
    x, y = np.mgrid[0:53.3:0.1, 0:120:0.1]
    locations = np.dstack((x, y))
    fq_frame_id_to_influence = analyze_frames(merged_df, groups.groups, locations)
    print('Produced analysis for %d frames' % len(fq_frame_id_to_influence))
    for fq_frame_id, influence in sorted(fq_frame_id_to_influence.items()):
        summarize_frame(fq_frame_id, influence)
        game_id = fq_frame_id[0]
        play_id = fq_frame_id[1]
        frame_id = fq_frame_id[2]
        filename = create_one_frame(game_id, play_id, frame_id, groups.get_group(fq_frame_id), x, y, influence,
                                    'images/2023/')
        print(filename)


def try_read_pff(filename):
    try:
        return pd.read_csv(filename)
    except pd.errors.EmptyDataError:
        return None


def main(argv):
    if len(argv) < 6:
        print('Usage: %s <games_csv> <plays_csv> <players_csv> <pff_csv> <tracking_csv>' %
              argv[0])
        return 1
    games_df = pd.read_csv(argv[1])
    plays_df = load_plays_data(argv[2])
    players_df = pd.read_csv(argv[3])
    pff_df = try_read_pff(argv[4])
    tracking_df = load_all_tracking_data(argv[5:])
    games_df, plays_df, tracking_df = standardize_all_dataframes(games_df, plays_df, tracking_df, pff_df=pff_df,
                                                                 players_df=players_df)
    merged_df = select_and_merge_data(games_df, plays_df, pff_df, players_df, tracking_df)
    analyze_all_plays(merged_df)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
