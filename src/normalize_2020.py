import pandas as pd


from standardize import normalize_column_formatting


def extract_games(unified_df):
    _GAMES_COLUMNS = ['gameId', 'gameWeather', 'homeTeamAbbr', 'humidity', 'location', 'season', 'stadium',
                      'stadiumType', 'temperature', 'turf', 'visitorTeamAbbr', 'week', 'windDirection', 'windSpeed']
    _COLUMN_REMAP = {}
    games_df = unified_df[_GAMES_COLUMNS].drop_duplicates().reset_index().drop(columns=['index'])
    games_df = games_df.rename(columns=_COLUMN_REMAP)
    return games_df


def extract_players(unified_df):
    _PLAYS_COLUMNS = ['nflId', 'playerHeight', 'playerWeight', 'playerCollegeName', 'position']
    _COLUMN_REMAP = {'playerHeight': 'height', 'playerWeight': 'weight', 'playerCollegeName': 'college',
                     'position': 'positionAbbr'}
    players_df = unified_df[_PLAYS_COLUMNS].drop_duplicates().reset_index().drop(columns=['index'])
    players_df = players_df.rename(columns=_COLUMN_REMAP)
    return players_df


def insert_midfield_distance(plays_df):
    midfield_index = plays_df[plays_df.yardlineSide.isnull()].index
    plays_df.loc[midfield_index, 'yardlineNumber'] = 50
    return plays_df


def rename_teams(plays_df):
    _RENAME = {'ARZ': 'ARI', 'BLT': 'BAL', 'CLV': 'CLE', 'HST': 'HOU'}
    for orig, update in _RENAME.items():
        idx = plays_df[plays_df.possessionTeam == orig].index
        plays_df.loc[idx, 'possessionTeam'] = update


def extract_plays(unified_df):
    _PLAYS_COLUMNS = ['defendersInTheBox', 'defensePersonnel', 'distance', 'down', 'fieldPosition', 'gameClock',
                      'gameId', 'homeScoreBeforePlay', 'offenseFormation', 'offensePersonnel',
                      'playId', 'possessionTeam', 'quarter', 'visitorScoreBeforePlay', 'yardLine', 'yards']
    _COLUMN_REMAP = {'defensePersonnel': 'personnelD', 'distance': 'yardsToGo',
                     'fieldPosition': 'yardlineSide', 'homeScoreBeforePlay': 'preSnapHomeScore',
                     'offensePersonnel': 'personnelO', 'visitorScoreBeforePlay': 'preSnapVisitorScore',
                     'yardLine': 'yardlineNumber'}
    plays_df = unified_df[_PLAYS_COLUMNS].drop_duplicates().reset_index().drop(columns=['index'])
    plays_df = plays_df.rename(columns=_COLUMN_REMAP)
    rename_teams(plays_df)
    plays_df['isSTPlay'] = False
    plays_df['specialTeamsPlayType'] = None
    plays_df['passResult'] = None
    plays_df = insert_midfield_distance(plays_df)
    return plays_df


def insert_football_frames(tracking_df):
    ball_carrier_df = tracking_df[tracking_df.nflId == tracking_df.nflIdRusher].copy()
    ball_carrier_df['team'] = 'football'
    ball_carrier_df['displayName'] = 'football'
    ball_carrier_df['nflId'] = None
    ball_carrier_df['jerseyNumber'] = None
    return tracking_df._append(ball_carrier_df, ignore_index=True).reset_index()


def extract_tracking(unified_df):
    _TRACKING_COLUMNS = ['a', 'dir', 'dis', 'displayName', 'gameId', 'jerseyNumber', 'nflId', 'playId', 's', 'team',
                         'x', 'y', 'nflIdRusher']
    tracking_df = unified_df[_TRACKING_COLUMNS].drop_duplicates().reset_index().drop(columns=['index'])
    tracking_df['event'] = 'handoff'
    tracking_df['frameId'] = 1
    tracking_df = insert_football_frames(tracking_df)
    return tracking_df


def normalize_2020_df(unified_df):
    unified_df = normalize_column_formatting(unified_df)
    games_df = extract_games(unified_df)
    players_df = extract_players(unified_df)
    plays_df = extract_plays(unified_df)
    tracking_df = extract_tracking(unified_df)
    return games_df, plays_df, players_df, tracking_df


def normalize_2020(filename):
    unified_df = pd.read_csv(filename, low_memory=False)
    return normalize_2020_df(unified_df)
