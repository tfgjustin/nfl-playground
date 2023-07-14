import abc
import math
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
import keras.backend as K

from plays import load_plays_data
from sklearn.model_selection import train_test_split
from standardize import standardize_tracking_dataframes, try_read_pff
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Add, AvgPool1D, AvgPool2D, BatchNormalization, Conv1D, Conv2D, Dense, \
    Dropout, Lambda, LayerNormalization, MaxPooling1D, MaxPooling2D
from keras.losses import Loss
from keras.optimizers import Adam
from keras.optimizers.schedules import LearningRateSchedule
from time import time
from tracking import load_all_tracking_data


def distance_from_ball(row):
    player_x = row['normX']
    player_y = row['normY']
    ball_x = row['ballPosX']
    ball_y = row['ballPosY']
    return math.sqrt((player_x - ball_x) ** 2 + (player_y - ball_y) ** 2)


def get_run_play_info(plays_df, tracking_df):
    # Get the info about plays which are not pass plays
    not_pass_plays_df = plays_df[plays_df.passResult.isnull()]
    handoffs_df = tracking_df[(tracking_df.event == 'handoff') & (tracking_df.team == 'football')][
        ['gameId', 'playId', 'frameId', 'normY']
    ].drop_duplicates().rename(columns={'frameId': 'handoffFrameId', 'normY': 'handoffY'})
    handoffs_df = pd.merge(handoffs_df, not_pass_plays_df, on=['gameId', 'playId'], how='inner')
    handoffs_df = handoffs_df[['gameId', 'playId', 'handoffFrameId', 'handoffY', 'possessionTeam']]
    run_end_df = tracking_df[(tracking_df.event == 'tackle') | (tracking_df.event == 'out_of_bounds')]
    run_end_df = run_end_df[run_end_df.team == 'football'][
        ['gameId', 'playId', 'frameId', 'normY']
    ].drop_duplicates().rename(columns={'frameId': 'runEndFrameId', 'normY': 'runEndY'})
    run_frames_df = pd.merge(handoffs_df, run_end_df, on=['gameId', 'playId'], how='inner')
    run_frames_df['runDistance'] = run_frames_df['runEndY'] - run_frames_df['handoffY']
    neg_distance_idx = run_frames_df[run_frames_df.runDistance < 0].index
    run_frames_df.loc[neg_distance_idx, 'runDistance'] = 0
    print('Found %d plays with handoffs and tackles/OOB' % len(run_frames_df))
    return run_frames_df


def get_ball_path(run_play_frames_df, tracking_df):
    ball_frames_df = tracking_df[tracking_df.team == 'football']
    ball_frames_df = pd.merge(run_play_frames_df, ball_frames_df, on=['gameId', 'playId'], how='left')
    ball_frames_df = ball_frames_df[(ball_frames_df.frameId >= ball_frames_df.handoffFrameId) & (
            ball_frames_df.frameId <= ball_frames_df.runEndFrameId)][
        ['gameId', 'playId', 'frameId', 'normX', 'normY', 'possessionTeam']
    ]
    ball_frames_df.rename(columns={'normX': 'ballPosX', 'normY': 'ballPosY'}, inplace=True)
    print('Found %d ball position frames' % len(ball_frames_df))
    return ball_frames_df


def infer_ball_carrier(run_play_frames_df, tracking_df):
    ball_frames_df = get_ball_path(run_play_frames_df, tracking_df)
    players_pos_df = pd.merge(ball_frames_df, tracking_df[tracking_df.team != 'football'],
                              on=['gameId', 'playId', 'frameId'], how='left')
    players_pos_df = players_pos_df[players_pos_df.team == players_pos_df.possessionTeam]
    players_pos_df['distanceFromBall'] = players_pos_df.apply(distance_from_ball, axis=1)
    player_distances_df = players_pos_df.groupby(
        ['gameId', 'playId', 'team', 'nflId']
    )['distanceFromBall'].mean().reset_index()
    # Only look at players who were within 1 yard of the ball during the run
    player_distances_df = player_distances_df[player_distances_df.distanceFromBall <= 1.0]
    # Now grab the rows which represent the players closest to the ball each play.
    player_distances_df.sort_values(by=['distanceFromBall'], ascending=[True], inplace=True)
    player_distances_df.drop_duplicates(subset=['gameId', 'playId'], inplace=True)
    player_distances_df.rename(columns={'nflId': 'ballCarrierId', 'team': 'ballCarrierTeam'}, inplace=True)
    player_distances_df = player_distances_df[['gameId', 'playId', 'ballCarrierId', 'ballCarrierTeam']]
    print('Found the closest %d players to the ball each time' % len(player_distances_df))
    return player_distances_df


def select_plays(plays_df, tracking_df):
    run_play_frames_df = get_run_play_info(plays_df, tracking_df)
    ball_carrier_df = infer_ball_carrier(run_play_frames_df, tracking_df)
    merged_df = pd.merge(ball_carrier_df, run_play_frames_df, on=['gameId', 'playId'], how='inner')
    return merged_df[['gameId', 'playId', 'ballCarrierId', 'ballCarrierTeam', 'handoffFrameId', 'runEndY']]


def construct_ball_carrier_array(df):
    ball_carrier_df = df[df.isBallCarrier][['normX', 'normY', 'xSpeed', 'ySpeed']]
    if len(ball_carrier_df) != 1:
        return None
    ball_carrier = np.repeat(ball_carrier_df.to_numpy(), 10).reshape(4, 10)
    base_carrier = np.vstack([np.zeros((2, 10)), ball_carrier, np.zeros((4, 10))])
    return np.tile(base_carrier, [11, 1]).reshape((11, 10, 10))


def construct_offense_array(df):
    offense_df = df[df.isOffense & ~df.isBallCarrier][['normX', 'normY', 'xSpeed', 'ySpeed']]
    if len(offense_df) != 10:
        return None
    # print(offense_df)
    # The offense is only relevant in four of the 10 rows for each defender, so we need to pad this out with zeros.
    base_offense = np.vstack([np.zeros((6, 10)), offense_df.to_numpy().transpose()])
    # Tile it on the first axis (basically stack it a bunch) and then reshape it to be the same as the defense.
    return np.tile(base_offense, [11, 1]).reshape((11, 10, 10))


def construct_defense_array(df):
    defense_df = df[~df.isOffense & (df.team != 'football')][['normX', 'normY', 'xSpeed', 'ySpeed']]
    if len(defense_df) != 11:
        return None
    defense_numpy = defense_df.to_numpy()
    # Copy it three times, but then cut off the first two columns since we only need two positions but three speeds.
    base_defense = np.tile(defense_numpy, 3)[:, 2:]
    # Now repeat each element 10 times (once for each offensive player) and reshape to the final sample frame.
    return np.repeat(base_defense, 10).reshape((11, 10, 10))


def create_distance_outputs(df):
    y = df['runEndY'].unique() - df[df.isBallCarrier]['normY'].unique()
    y = round(max([-99, min([99, y[0]])]))
    y_idx = int(y + 99)
    y_array = np.zeros(199, dtype=np.float32)
    y_array[y_idx] = 1
    return y_array


def construct_mirror_sample(df):
    mirror_df = df.copy()
    # Reverse everything along the X-axis to create a mirrored sample
    mirror_df['xSpeed'] = -mirror_df['xSpeed']
    mirror_df['xAccel'] = -mirror_df['xAccel']
    mirror_df['xDirectionFactor'] = -mirror_df['xDirectionFactor']
    mirror_df['normX'] = 53.3 - mirror_df['normX']
    mirror_df['ballStartX'] = 53.3 - mirror_df['ballStartX']
    return mirror_df


def construct_one_sample(df):
    # Input:
    # The rusher is R
    # For each defender D_j:
    # - For each player on offense O_k:
    #   (0&1) The Sx and Sy values for D_j
    #   (2&3) R(x) - D_j(x), and R(y) - D_j(y)
    #   (4&5) R(Sx) - D_j(Sx), and R(Sy) - D_j(Sy)
    #   (6&7) O_k(x) - D_j(x), and O_k(y) - D_j(y)
    #   (8&9) O_k(Sx) - D_j(Sx), and O_k(Sy) - D_j(Sy)
    ball_carrier_df = construct_ball_carrier_array(df)
    if ball_carrier_df is None:
        return None
    offense_df = construct_offense_array(df)
    if offense_df is None:
        return None
    defense_df = construct_defense_array(df)
    if defense_df is None:
        return None
    X = defense_df - ball_carrier_df - offense_df
    y = create_distance_outputs(df)
    return [y, X]


def construct_samples_from_play(_, rows):
    df = rows.reset_index()
    mirror_df = construct_mirror_sample(df)
    regular_xy = construct_one_sample(df)
    if regular_xy is None:
        return None
    output = [regular_xy]
    mirror_xy = construct_one_sample(mirror_df)
    if mirror_xy is not None:
        output.append(mirror_xy)
    return output


def construct_training_df(run_play_info_df, tracking_df):
    np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: '%.3g' % x))
    df = pd.merge(run_play_info_df, tracking_df, on=['gameId', 'playId'], how='left')
    # Get one frame on either side of the handoff (i.e., within 0.1s). This will roughly 3x the amount of data we have
    # without (hopefully) affecting the quality of the predictions too much.
    df = df[abs(df.frameId - df.handoffFrameId) <= 1]
    df['isOffense'] = df['ballCarrierTeam'] == df['team']
    df['isBallCarrier'] = df['ballCarrierId'] == df['nflId']
    data = list()
    for game_play_id, group in df.groupby(['gameId', 'playId', 'frameId']):
        samples = construct_samples_from_play(game_play_id, group)
        if samples is not None:
            data.extend(samples)
    print('Number of frames found:', len(data))
    return data


class MyLearningRateSchedule(LearningRateSchedule):
    # One cycle scheduler over a total of 50 epochs for each fit with lower lr being 0.0005 and upper lr being 0.001
    def __init__(self, initial_rate=1e-3, minimum_rate=1e-4, num_steps=50):
        self._initial_rate = initial_rate
        self._minimum_rate = minimum_rate
        self._num_steps = num_steps

    def get_config(self):
        config = {
            '_initial_rate': self._initial_rate,
            '_minimum_rate': self._minimum_rate,
            '_num_steps': self._num_steps
        }
        return config

    @abc.abstractmethod
    def __call__(self, step):
        steps_to_min = self._num_steps - step
        return tf.cond(
            steps_to_min <= 0,
            true_fn=lambda: self._minimum_rate,
            false_fn=lambda: self._minimum_rate + (
                    (self._initial_rate - self._minimum_rate) * (steps_to_min / self._num_steps)
            )
        )


class CrpsLoss(Loss):
    def __init__(self):
        super(CrpsLoss, self).__init__()

    def call(self, y_true, y_pred):
        # Create a CDF for both the observations and the predictions
        sum_true = tf.math.cumsum(y_true, axis=1)
        sum_true = tf.clip_by_value(sum_true, 0, 1)
        sum_pred = tf.math.cumsum(y_pred, axis=1)
        sum_pred = tf.clip_by_value(sum_pred, 0, 1)
        # Get the square of the error and divide it by the total number of data points
        raw_error = sum_true - sum_pred
        square_error = raw_error * raw_error
        # y_pred.shape[0] is None because it's not known until execution time, so we get it from the Keras backend
        batch_size = tf.cast(K.shape(y_pred)[0], tf.float32)
        num_columns = tf.cast(y_pred.shape[-1], tf.float32)
        return tf.reduce_sum(square_error) / tf.math.multiply(num_columns, batch_size)


def create_model():
    input_dense_players = Input(shape=(11, 10, 10), name='numerical_players_feature_input')
    # Convolutional layers to detect interactions between offense, defense, and the rusher
    x = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), activation=None)(input_dense_players)
    x = Activation('relu')(x)
    x = Conv2D(160, kernel_size=(1, 1), strides=(1, 1), activation=None)(x)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), activation=None)(x)
    x = Activation('relu')(x)
    # Max pooling will find things like the closest/furthest away player from each defender
    # whereas average pooling will find roughly where defenders are in relation to everyone else.
    x_max = MaxPooling2D(pool_size=(1, 10))(x)
    x_avg = AvgPool2D(pool_size=(1, 10))(x)
    # Bias mostly towards the average, but give some weight to the extremes.
    x_max = Lambda(lambda x1: x1 * 0.3)(x_max)
    x_avg = Lambda(lambda x1: x1 * 0.7)(x_avg)
    x = Add()([x_max, x_avg])
    x = BatchNormalization()(x)
    # Squeeze it down to one tensor for each defender.
    x = Lambda(lambda y: tf.squeeze(y, 2))(x)
    x = Conv1D(160, kernel_size=1, strides=1, activation=None)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(96, kernel_size=1, strides=1, activation=None)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(96, kernel_size=1, strides=1, activation=None)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    # Again, look for a mix of extremes and the averages w.r.t. each defender.
    x_max = MaxPooling1D(pool_size=11)(x)
    x_max = Lambda(lambda x1: x1 * 0.3)(x_max)
    x_avg = AvgPool1D(pool_size=11)(x)
    x_avg = Lambda(lambda x1: x1 * 0.7)(x_avg)
    x = Add()([x_max, x_avg])
    x = Lambda(lambda y: tf.squeeze(y, 1))(x)
    x = Dense(96)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Activation('relu')(x)
    x = LayerNormalization()(x)
    x = Dropout(0.5)(x)
    # x = BatchNormalization()(x)
    per_yard_prob = Dense(199, activation='softmax', name='per_yard_prob')(x)
    model = Model(inputs=[input_dense_players], outputs=[per_yard_prob])
    learning_rate = MyLearningRateSchedule()
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=CrpsLoss())
    print(model.summary())
    return model


def split_x_y(data):
    y_yard_distribution = np.array([d[0] for d in data])
    X = np.array([d[1] for d in data])
    return X, y_yard_distribution


def train_model(training_data, model):
    data_train, data_test = train_test_split(training_data, test_size=0.2, random_state=47)
    data_train, data_valid = train_test_split(data_train, test_size=0.25, random_state=47)
    print('Train:', len(data_train), 'Test:', len(data_test))
    X_train, y_train_yard_distribution = split_x_y(data_train)
    X_valid, y_valid_yard_distribution = split_x_y(data_valid)
    y_valid = {'per_yard_prob': y_valid_yard_distribution},
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, min_delta=1e-3, restore_best_weights=True)
    model.fit(X_train, y={'per_yard_prob': y_train_yard_distribution}, validation_data=(X_valid, y_valid),
              batch_size=64, epochs=150, shuffle=True, verbose=1, callbacks=[early_stopping])
    print('\n\nTraining done\n')
    X_test, y_test_yard_distribution = split_x_y(data_test)
    model.evaluate(x=X_test, y={'per_yard_prob': y_test_yard_distribution})
    y_pred = model.predict(X_test)
    yards = np.arange(-99, 100)
    for idx in range(len(y_pred)):
        expected_yards = np.sum(y_pred[idx] * yards)
        actual_yards = np.sum(y_test_yard_distribution[idx] * yards)
        diff = expected_yards - actual_yards
        print('%04d Expected: %5.2f Actual: %5.2f Diff: %5.2f' % (idx, expected_yards, actual_yards, diff))
    model.save('expected_yards.%f.model' % time())


def main(argv):
    if len(argv) < 6:
        print('Usage: %s <games_csv> <plays_csv> <players_csv> <pff_csv> <tracking_csv>' %
              argv[0])
        return 1
    games_df = pd.read_csv(argv[1])
    plays_df = load_plays_data(games_df, argv[2])
    players_df = pd.read_csv(argv[3])
    pff_df = try_read_pff(argv[4])
    tracking_df = load_all_tracking_data(argv[5:])
    tracking_df = standardize_tracking_dataframes(games_df, plays_df, tracking_df, pff_df=pff_df, players_df=players_df)
    run_play_info_df = select_plays(plays_df, tracking_df)
    training_data = construct_training_df(run_play_info_df, tracking_df)
    model = create_model()
    train_model(training_data, model)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
