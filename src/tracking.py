import pandas as pd


def is_football(row):
    return row['team'] == 'football'


def normalize_column_names(tracking_df):
    """
    Normalize everything to what we see in the 2023 Big Data Bowl.

    The only other data I currently have is from ~2018-2019, so there may be gaps.
    :param tracking_df: DataFrame with full tracking data.
    :return: DataFrame with standardized column names.
    """
    return tracking_df.rename(columns={'frame.id': 'frameId'}, errors='ignore')


def maybe_rename_football(tracking_df):
    """
    Standardize the name of the football in the tracking DataFrame.

    :param tracking_df: DataFrame with full tracking data
    :return: DataFrame with a standardized name for the football
    """
    ball_rows_idx = tracking_df[tracking_df.team == 'ball'].index
    if ball_rows_idx.empty:
        return tracking_df
    tracking_df.loc[ball_rows_idx, 'team'] = 'football'
    return tracking_df


def maybe_insert_acceleration(tracking_df):
    """ If acceleration data is not present, add in zeros.

    :param tracking_df: DataFrame with full tracking data (ball and players), which may or may not have an 'a' column
    :return: DataFrame guaranteed to have an acceleration ('a') column
    """
    if 'a' not in tracking_df:
        tracking_df['a'] = 0.0
    return tracking_df


def replace_event_none_strings_with_none(tracking_df):
    """
    Many events are tagged in the data as 'None' (the string) instead of None (the value). Fix that.

    :param tracking_df: DataFrame with player and ball tracking data.
    :return: DataFrame where the events have None values instead of 'None' strings
    """
    idx = tracking_df[tracking_df.event == 'None'].index
    if idx.empty:
        return tracking_df
    tracking_df.loc[idx, 'event'] = None
    return tracking_df


def filter_football(tracking_df):
    """
    Removes the football from the tracking data.

    :param tracking_df: raw DataFrame from CSV files
    :return: DataFrame with football rows removed.
    """
    return tracking_df[tracking_df.team != 'football']


def group_and_sort_events_by_frames(player_tracking_df):
    """
    Group events from player-only (i.e., no football) tracking data keyed by (game, play, frame, event) and count rows.

    :param player_tracking_df: DataFrame with football-based rows removed
    :return: DataFrameGroupBy sorted by (game, play, frame, event)
    """
    player_tracking_df = player_tracking_df[player_tracking_df.event != 'None']
    groupby_df = player_tracking_df[['gameId', 'playId', 'frameId', 'event', 'nflId']].groupby(
            ['gameId', 'playId', 'frameId', 'event']
            ).count().reset_index()
    return groupby_df.sort_values(by=['gameId', 'playId', 'frameId', 'event'], ascending=[True, True, True, True])


def find_best_event_frames(events_df):
    """
    Find the 'best' event frame for a given (game, play, event) tuple.

    The 'best' frame for a (game, play, event) key is the one which has the most players associated with it, and
    happened earliest in the play. This is purely a heuristic which may be broken, but it does deal with cases where
    an event is smeared across multiple players. E.g., the ball_snap event is associated with 17 players on frame 4, and
    the other 5 players on frame 5. This will return frame 4.

    :param events_df: DataFrameGroupBy returned from group_and_sort_events_by_frames
    :return: DataFrame sorted by game, play, and frameId, with the associated event
    """
    event_frames_df = pd.DataFrame({
        'gameId': pd.Series(dtype='int64'),
        'playId': pd.Series(dtype='int64'),
        'frameId': pd.Series(dtype='int64'),
        'event': pd.Series(dtype='str')})
    for name, group in events_df.groupby(['gameId', 'playId', 'event']):
        if len(group) == 1:
            event_frames_df = event_frames_df.append(group[['gameId', 'playId', 'frameId', 'event']],
                                                     ignore_index=True)
        else:
            best_row = group.sort_values(by=['nflId', 'frameId'], ascending=[False, True])
            event_frames_df = event_frames_df.append(best_row.iloc[0][['gameId', 'playId', 'frameId', 'event']],
                                                     ignore_index=True)
    return event_frames_df.sort_values(
        by=['gameId', 'playId', 'frameId'], ascending=[True, True, True]
    )


def create_unique_event_frame_index(player_tracking_df):
    """
    Given player-only tracking data, finds the best frame within a play for a given event.

    :param player_tracking_df: DataFrame with player-only tracking data
    :return: DataFrame sorted by game, play, and frameId, with the associated event
    """
    events_df = group_and_sort_events_by_frames(player_tracking_df)
    return find_best_event_frames(events_df)


def create_tracking_with_unique_play_events(tracking_df):
    """
    Cleans up tracking DataFrame so each event in a play happens only once and is applied to all entities on the field
    at the same time.

    This function is idempotent, and can safely be called multiple times on the same data.

    :param tracking_df: DataFrame with full motion tracking data (players and ball)
    :return: DataFrame where each event in a play happens only once and at the same time for everyone
    """
    player_tracking_df = filter_football(tracking_df)
    event_frame_index_df = create_unique_event_frame_index(player_tracking_df)
    # Now drop the existing events and join in the best events
    tracking_df = tracking_df.drop(columns=['event'])
    clean_tracking_df = tracking_df.merge(event_frame_index_df, on=['gameId', 'playId', 'frameId'], how='left')
    return clean_tracking_df.sort_index(axis=1).sort_index(axis=0)


def load_all_tracking_data(files):
    """
    Load all the tracking data from one or more files and ensure everything is standardized/looks the same.

    :param files: Iterable with a name of files containing tracking data
    :return: one DataFrame with standardized tracking data, sorted by column names and plays.
    """
    tracking_df = pd.concat([pd.read_csv(f) for f in files]).reset_index()
    tracking_df = normalize_column_names(tracking_df)
    tracking_df = maybe_rename_football(tracking_df)
    tracking_df = maybe_insert_acceleration(tracking_df)
    tracking_df = replace_event_none_strings_with_none(tracking_df)
    tracking_df = create_tracking_with_unique_play_events(tracking_df)
    return tracking_df.sort_index(axis=1).sort_index(axis=0)
