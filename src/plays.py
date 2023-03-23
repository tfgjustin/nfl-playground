import numpy as np
import pandas as pd


def is_special_teams_play(row):
    return row['isSTPlay']


def maybe_insert_yards_after_catch(plays_df):
    """
    Ensure that the play-by-play DataFrame has a column for yardsAfterCatch, even if it's just NaN

    :param plays_df: DataFrame with play-by-play data
    :return: DataFrame with play-by-play data guaranteed to have a yardsAfterCatch column
    """
    if 'yardsAfterCatch' in plays_df:
        return plays_df
    plays_df['yardsAfterCatch'] = np.nan
    return plays_df


def calculate_absolute_yardline_number(row):
    """
    Calculate the absolute yardline number from play-by-play data

    :param row: one row (i.e., one play) from the play-by-play DataFrame
    :return: the distance from the goal endzone
    """
    if row['possessionTeam'] == row['yardlineSide']:
        # The team is on their own side of the field. The yardline is the number of yards from their own goal.
        return 100 - row['yardlineNumber']
    else:
        # They are on the opponent's side of the field, so the yardline is the distance from the opponent's goal
        return row['yardlineNumber']


def maybe_insert_absolute_yardline_number(plays_df):
    """
    Ensure that the play-by-play DataFrame has the absoluteYardline (i.e., distance from the goal endzone)

    :param plays_df: DataFrame with play-by-play data
    :return: DataFrame guaranteed to have absoluteYardlineNumber populated
    """
    if 'absoluteYardlineNumber' in plays_df:
        return plays_df
    plays_df['absoluteYardlineNumber'] = plays_df.apply(lambda row: calculate_absolute_yardline_number(row), axis=1)
    return plays_df


def normalize_game_clock(plays_df):
    """
    Standardize the game clock to be just MM:SS

    :param plays_df: DataFrame with play-by-play data
    :return: DataFrame with the gameClock only with MM:SS, not MM:SS:00
    """
    # Some game clocks are in MM:SS and others are in MM:SS:xx where xx appears to always be 00.
    # Cut the game clock to MM:SS
    plays_df['gameClock'] = plays_df['gameClock'].str[:5]
    return plays_df


def insert_defensive_team_abbreviation(row):
    if row['possessionTeam'] == row['homeTeamAbbr']:
        return row['visitorTeamAbbr']
    else:
        return row['homeTeamAbbr']


def maybe_add_defensive_team_abbreviations(games_df, plays_df):
    if 'defensiveTeam' in plays_df:
        return plays_df
    # First, figure out the abbreviations of the home and visitor teams
    game_teams_df = games_df[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']]
    play_teams_df = plays_df[['gameId', 'playId', 'possessionTeam']]
    merged_df = pd.merge(play_teams_df, game_teams_df, on=['gameId'], how='left')
    merged_df['defensiveTeam'] = merged_df.apply(insert_defensive_team_abbreviation, axis=1)
    merged_df.drop(columns=['homeTeamAbbr', 'visitorTeamAbbr', 'possessionTeam'], inplace=True)
    plays_df = pd.merge(plays_df, merged_df, on=['gameId', 'playId'], how='left')
    return plays_df


def load_plays_data(games_df, plays_csv):
    """
    Load all the play-by-play data and ensure everything is standardized/looks the same.

    :param games_df: DataFrame containing the list of games and information about those games
    :param plays_csv: name of a file containing play-by-play data
    :return: one DataFrame with standardized play-by-play data, sorted by column names and plays.
    """
    plays_df = pd.read_csv(plays_csv)
    plays_df = plays_df.rename(columns={
        'GameClock': 'gameClock',
        'PassResult': 'passResult',
        'personnel.defense': 'personnelD',
        'personnel.offense': 'personnelO',
        'HomeScoreBeforePlay': 'preSnapHomeScore',
        'VisitorScoreBeforePlay': 'preSnapVisitorScore',
        'YardsAfterCatch': 'yardsAfterCatch'
    }, errors='ignore')
    plays_df = maybe_insert_yards_after_catch(plays_df)
    plays_df = maybe_insert_absolute_yardline_number(plays_df)
    plays_df = normalize_game_clock(plays_df)
    plays_df = maybe_add_defensive_team_abbreviations(games_df, plays_df)
    plays_df.sort_values(by=['gameId', 'playId', 'gameClock'], ascending=[True, True, False], inplace=True)
    return plays_df.sort_index(axis=0)
