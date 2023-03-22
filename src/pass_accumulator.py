import numpy as np
import pandas as pd

from collections import defaultdict, deque
from plays import is_special_teams_play


_DEFAULT_DISTANCE_BUCKETS = [0, 5, 10, 20, 40, 99]
_PASS_STATUS_TO_NAME = {'IN': 'Int', 'R': 'Scramble', 'S': 'Sack'}


class TeamPassHistoryRepository(object):
    """
    A repository for the history of passing attempts by a given in a game.

    Keeps track of attempts, completions, and other outcomes (e.g., sacks, interceptions), on both game-level and within
    a certain lookback number of plays on offense.
    """

    def __init__(self, lookback_length=15, distance_buckets=None):
        """
        Constructor. Can optionally specify the size of the "lookback" window for what is considered "recent".

        :param lookback_length: Number of plays on offense to consider "recent".
        :param distance_buckets: Number of yards downfield as a boundary to bucket attempts. E.g., a value of '0' will
        create a bucket for all plays behind the line of scrimmage, and another bucket of '5' will contain all passes
        between 0 and 4 yards in length.
        """
        self._distance_buckets = _DEFAULT_DISTANCE_BUCKETS if distance_buckets is None else distance_buckets
        # Recent attempts and completions
        self._recent_pass_attempt_by_distance = {d: deque(list(), lookback_length) for d in self._distance_buckets}
        self._recent_pass_complete_by_distance = {d: deque(list(), lookback_length) for d in self._distance_buckets}
        self._recent_pass_other_by_status = {s: deque(list(), lookback_length) for s in _PASS_STATUS_TO_NAME.keys()}
        # Total attempts and completions
        self._total_pass_attempt_by_distance = {d: 0 for d in self._distance_buckets}
        self._total_pass_complete_by_distance = {d: 0 for d in self._distance_buckets}
        self._total_pass_other_by_status = {s: 0 for s in _PASS_STATUS_TO_NAME.keys()}
        self._total_plays = 0

    def total_plays(self):
        """
        :return: The total number of plays on offense this game.
        """
        return self._total_plays

    def add_play(self, pass_info):
        """
        Add a (non-special-teams) offensive play to the repository.

        :param pass_info: A tuple of (pass status, distance bucket)
        :return: None
        """
        self._total_plays += 1
        if pass_info is None:
            self.add_not_a_pass()
        else:
            self.add_pass(pass_info[0], pass_info[1])

    def add_not_a_pass(self):
        """
        This play was not a pass, so update all state when a pass was not attempted.

        :return: None
        """
        self.update_recent_passes(None, None)

    def add_pass(self, pass_status, distance_bucket):
        """
        Add a pass play to the history repository and update the relevant state.

        :param pass_status: indicates if the pass was complete, incomplete, intercepted, or if the QB ran/was sacked
        :param distance_bucket: the value of the boundary of the bucket, or None if the pass did not leave the QB's hand
        :return: None
        """
        if pass_status in ('C', 'I'):
            self.add_normal_pass(pass_status, distance_bucket)
        elif pass_status in _PASS_STATUS_TO_NAME:
            self.add_other_pass(pass_status)
        else:
            print('Unknown pass status: ', pass_status)

    def add_normal_pass(self, pass_status, distance_bucket):
        """
        Add information about a pass which actually left the QB's hand.

        :param pass_status: indicates if the pass was complete, incomplete, intercepted, or if the QB ran/was sacked
        :param distance_bucket: the value of the boundary of the bucket, or None if the pass did not leave the QB's hand
        :return: None
        """
        was_complete = 1 if pass_status == 'C' else 0
        self._total_pass_attempt_by_distance[distance_bucket] += 1
        self._total_pass_complete_by_distance[distance_bucket] += was_complete
        self.update_recent_passes(distance_bucket, was_complete)

    def add_other_pass(self, pass_status):
        """
        Add information about a pass which did not leave the QB's hand.

        :param pass_status: indicates why the pass did not leave the QB's hand (sack, scramble) or if it was intercepted
        :return: None
        """
        if pass_status in self._total_pass_other_by_status:
            self._total_pass_other_by_status[pass_status] += 1
        self.update_recent_passes(pass_status, 0)

    def update_recent_passes(self, distance_bucket, was_complete):
        """
        Update the lookback windows of recent passes. This needs to happen on all non-special teams plays.

        :param distance_bucket: the value of the boundary of the bucket, or None if the pass did not leave the QB's hand
        :param was_complete: if the pass (if any) was completed.
        :return: None
        """
        for bucket in self._recent_pass_attempt_by_distance:
            self._recent_pass_attempt_by_distance[bucket].append(1 if bucket == distance_bucket else 0)
            self._recent_pass_complete_by_distance[bucket].append(was_complete if bucket == distance_bucket else 0)
        for status in _PASS_STATUS_TO_NAME.keys():
            self._recent_pass_other_by_status[status].append(1 if status == distance_bucket else 0)

    def generate_df_row(self, game_id, play_id):
        """
        Generate a row used by the DataFrame which is input to our predictive model.

        :param game_id: NFL identifier of the game
        :param play_id: NFL identifier of the play
        :return: a Python list with all the values (instead of a DataFrame row since this is more efficient)
        """
        output = []
        output.extend([self._total_pass_attempt_by_distance[bucket] for bucket in sorted(self._distance_buckets)])
        output.extend([self._total_pass_complete_by_distance[bucket] for bucket in sorted(self._distance_buckets)])
        output.extend([self._total_pass_other_by_status[status] for status in sorted(_PASS_STATUS_TO_NAME.keys())])
        output.extend([sum(self._recent_pass_attempt_by_distance[bucket]) for bucket in sorted(self._distance_buckets)])
        output.extend([
            sum(self._recent_pass_complete_by_distance[bucket]) for bucket in sorted(self._distance_buckets)
        ])
        output.extend([
            sum(self._recent_pass_other_by_status[status]) for status in sorted(_PASS_STATUS_TO_NAME.keys())
        ])
        return [game_id, play_id, self._total_plays] + output


class PassAccumulator(object):
    def __init__(self, games_df, plays_df, lookback_length=15, distance_buckets=None):
        self._games_df = games_df
        self._plays_df = plays_df
        self._lookback_length = lookback_length
        self._distance_buckets = _DEFAULT_DISTANCE_BUCKETS if distance_buckets is None else distance_buckets
        self._current_game_id = None
        self._team_history = None
        self._team_abbr_by_game = self.create_abbreviation_mapping()
        self._df_columns = self.create_columns()
        self._output_data = list()

    def create_columns(self):
        all_attempt_columns = ['allPassAttLe%02dYds' % y for y in self._distance_buckets]
        all_complete_columns = ['allPassCompLe%02dYds' % y for y in self._distance_buckets]
        all_other_columns = ['allPass%s' % s for s in sorted(_PASS_STATUS_TO_NAME.keys())]
        all_columns = all_attempt_columns + all_complete_columns + all_other_columns
        recent_attempt_columns = ['recentPassAttLe%02dYds' % y for y in self._distance_buckets]
        recent_complete_columns = ['recentPassCompLe%02dYds' % y for y in self._distance_buckets]
        recent_other_columns = ['recentPass%s' % s for s in sorted(_PASS_STATUS_TO_NAME.keys())]
        recent_columns = recent_attempt_columns + recent_complete_columns + recent_other_columns
        return ['gameId', 'playId', 'totalPlays'] + all_columns + recent_columns

    def create_abbreviation_mapping(self):
        abbr_dict = defaultdict(dict)
        for _, row in self._games_df.iterrows():
            abbr_dict[row['gameId']]['homeTeam'] = row['homeTeamAbbr']
            abbr_dict[row['gameId']]['awayTeam'] = row['visitorTeamAbbr']
        return abbr_dict

    def annotate(self):
        # TODO: Make sure we sort by gameId and then playId
        for _, row in self._plays_df.iterrows():
            # print(row)
            self.parse_one_play(row)

    def df(self):
        return pd.DataFrame(self._output_data, columns=self._df_columns)

    def parse_one_play(self, row):
        self.maybe_init_new_game(row)
        if is_special_teams_play(row):
            return
        poss_team = row['possessionTeam']
        assert poss_team in self._team_history
        pass_info = self.pass_status_and_distance_bucket(row)
        # print(poss_team, pass_info)
        self._team_history[poss_team].add_play(pass_info)
        # Give it time to warm up so the lookback length is meaningful.
        if self._team_history[poss_team].total_plays() >= self._lookback_length:
            self._output_data.append(self._team_history[poss_team].generate_df_row(row['gameId'], row['playId']))

    def maybe_init_new_game(self, row):
        game_id = row['gameId']
        if self._current_game_id is not None and self._current_game_id == game_id:
            return
        self._current_game_id = game_id
        assert game_id in self._team_abbr_by_game
        home_team_abbr = self._team_abbr_by_game[game_id]['homeTeam']
        away_team_abbr = self._team_abbr_by_game[game_id]['awayTeam']
        self._team_history = dict()
        self._team_history[home_team_abbr] = TeamPassHistoryRepository(lookback_length=self._lookback_length,
                                                                       distance_buckets=self._distance_buckets)
        self._team_history[away_team_abbr] = TeamPassHistoryRepository(lookback_length=self._lookback_length,
                                                                       distance_buckets=self._distance_buckets)

    def pass_status_and_distance_bucket(self, row):
        complete_status = row['passResult'] if 'passResult' in row else np.nan
        if pd.isna(complete_status):
            # Not a pass.
            return None
        if complete_status in _PASS_STATUS_TO_NAME:
            # They got sacked, it turned into a run play, or was intercepted.
            return [complete_status, None]
        actual_distance = row['PassLength']
        if pd.isna(actual_distance):
            # Some other odd outcome
            return None
        for distance in sorted(self._distance_buckets):
            if actual_distance < distance:
                return [complete_status, distance]
        # It should not be possible to have pass longer than 99 yards
        return None
