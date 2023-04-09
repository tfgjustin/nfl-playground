import abc
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

from normalize_2020 import normalize_2020_df
from plays import normalize_plays_data
from sklearn.model_selection import KFold

from standardize import standardize_tracking_dataframes
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import Add, AvgPool1D, AvgPool2D, BatchNormalization, Conv1D, Conv2D, Dense, \
    Dropout, Lambda, LayerNormalization, MaxPooling1D, MaxPooling2D
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from time import time


def load_data(filename):
    df_train = pd.read_csv(filename, dtype={'WindSpeed': 'object'})
    print(len(df_train))
    return df_train


def select_plays(plays_df, tracking_df):
    run_end_df = plays_df[['gameId', 'playId', 'absoluteYardlineNumber', 'yards']].copy()
    run_end_df['runEndY'] = 110 - run_end_df['absoluteYardlineNumber'] + run_end_df['yards']
    run_end_df['runDistance'] = run_end_df['yards']
    ball_carrier_df = tracking_df[tracking_df.nflId == tracking_df.nflIdRusher].copy()
    ball_carrier_df['ballCarrierId'] = ball_carrier_df['nflIdRusher']
    ball_carrier_df['ballCarrierTeam'] = ball_carrier_df['team']
    ball_carrier_df['targetFrameId'] = 1
    merged_df = pd.merge(run_end_df, ball_carrier_df, on=['gameId', 'playId'], how='inner')
    merged_df['isOffense'] = merged_df['ballCarrierTeam'] == merged_df['team']
    merged_df['isBallCarrier'] = merged_df['ballCarrierId'] == merged_df['nflId']
    return merged_df[
        ['gameId', 'playId', 'isOffense', 'isBallCarrier', 'targetFrameId', 'runEndY', 'runDistance']
    ].reset_index()


def construct_ball_carrier_array(df):
    ball_carrier_df = df[df.isBallCarrier][['xSpeed', 'ySpeed', 'normX', 'normY']]
    if len(ball_carrier_df) != 1:
        return None
    ball_carrier = np.repeat(ball_carrier_df.to_numpy(), 10).reshape(4, 10)
    base_carrier = np.vstack([np.zeros((4, 10)), ball_carrier, np.zeros((2, 10))])
    return np.tile(base_carrier.transpose(), [11, 1]).reshape((11, 10, 10))


def construct_offense_array(df):
    offense_df = df[df.isOffense & ~df.isBallCarrier][['xSpeed', 'ySpeed', 'normX', 'normY']]
    if len(offense_df) != 10:
        return None
    offense_np = offense_df.to_numpy()
    # Shuffle the data, so each time we load it we provide a new view of it to the convolution layers.
    np.random.shuffle(offense_np)
    # The offense is only relevant in four of the 10 rows for each defender, so we need to pad this out with zeros.
    base_offense = np.vstack([offense_np.transpose(), np.zeros((6, 10))])
    # Tile it on the first axis (basically stack it a bunch) and then reshape it to be the same as the defense.
    return np.tile(base_offense.transpose(), [11, 1]).reshape((11, 10, 10))


def construct_defense_array(df):
    defense_df = df[~df.isOffense & (df.team != 'football')][['xSpeed', 'ySpeed', 'normX', 'normY']]
    if len(defense_df) != 11:
        return None
    defense_numpy = defense_df.to_numpy()
    # Shuffle the data, so each time we load it we provide a new view of it to the convolution layers.
    np.random.shuffle(defense_numpy)
    negative_defense_numpy = -1 * defense_numpy
    # The first one is negative since it'll be the offense minus the defense.
    # The second two are positive because it's the defense minus the rusher, and then just the defense.
    base_defense = np.hstack([negative_defense_numpy, negative_defense_numpy, defense_numpy])[:, :10]
    # Now tile each element 10 times (once for each offensive player) and reshape to the final sample frame.
    return np.tile(base_defense, 10).reshape((11, 10, 10))


def create_distance_outputs(df):
    y_array = np.zeros(80, dtype=np.float32)
    if 'runDistance' not in df:
        return y_array
    y = df['runDistance'].unique()
    y = round(max([-29, min([50, y[0]])]))
    y_idx = int(y + 29)
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


def construct_one_sample(play_id, df):
    season = int(str(play_id)[:4])
    # Input:
    # The rusher is R
    # For each defender D_j:
    # - For each player on offense O_k:
    #   (2&3) R(x) - D_j(x), and R(y) - D_j(y)
    #   (4&5) R(Sx) - D_j(Sx), and R(Sy) - D_j(Sy)
    #   (6&7) O_k(x) - D_j(x), and O_k(y) - D_j(y)
    #   (8&9) O_k(Sx) - D_j(Sx), and O_k(Sy) - D_j(Sy)
    #   (0&1) The Sx and Sy values for D_j
    ball_carrier_df = construct_ball_carrier_array(df)
    if ball_carrier_df is None:
        return None
    offense_df = construct_offense_array(df)
    if offense_df is None:
        return None
    defense_df = construct_defense_array(df)
    if defense_df is None:
        return None
    X = defense_df + ball_carrier_df + offense_df
    y = create_distance_outputs(df)
    return [y, X, season]


def construct_samples_from_play(_, game_play_id, rows, create_mirror_samples=True):
    df = rows.sort_values(by=['nflId']).reset_index()
    regular_xy = construct_one_sample(game_play_id[1], df)
    if regular_xy is None:
        return None
    if np.isnan(np.sum(regular_xy[1])):
        print('NaN in ', game_play_id)
        print(df.info())
        print(df.head(23))
        return None
    output = [regular_xy]
    if not create_mirror_samples:
        return output
    mirror_df = construct_mirror_sample(df)
    mirror_xy = construct_one_sample(game_play_id[1] + 6000, mirror_df)
    if mirror_xy is not None:
        output.append(mirror_xy)
    return output


def construct_training_df(run_play_info_df, tracking_df, create_mirror_samples=True, samples_per_play=3):
    np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: '%.6g' % x))
    df = pd.merge(run_play_info_df, tracking_df, on=['gameId', 'playId'], how='left')
    # Get one frame on either side of the handoff (i.e., within 0.1s). This will roughly 3x the amount of data we have
    # without (hopefully) affecting the quality of the predictions too much.
    df = df[abs(df.frameId - df.targetFrameId) <= 1]
    data = list()
    n = 0
    for game_play_id, group in df.groupby(['gameId', 'playId', 'frameId']):
        for _ in range(samples_per_play):
            samples = construct_samples_from_play(n, game_play_id, group, create_mirror_samples=create_mirror_samples)
            if samples is None:
                break
            data.extend(samples)
        n += 1
        if n % 1000 == 0:
            print('%5d %s' % (n, game_play_id[1]))
    print('Number of frames found:', len(data))
    return data


def split_x_y(data):
    y_yard_distribution = np.array([d[0] for d in data])
    X = np.array([d[1] for d in data])
    season = np.array([d[2] for d in data])
    return X, y_yard_distribution, season


def create_training_inputs(unified_df, create_mirror_samples=True, samples_per_play=3):
    games_df, plays_df, players_df, tracking_df = normalize_2020_df(unified_df)
    plays_df = normalize_plays_data(games_df, plays_df)
    tracking_df = standardize_tracking_dataframes(games_df, plays_df, tracking_df, players_df=players_df)
    run_play_info_df = select_plays(plays_df, tracking_df)
    training_data = construct_training_df(run_play_info_df, tracking_df, create_mirror_samples=create_mirror_samples,
                                          samples_per_play=samples_per_play)
    return split_x_y(training_data)


class MyLearningRateSchedule(LearningRateSchedule):
    def __init__(self, _initial_rate=1e-3, _minimum_rate=5e-4, _num_steps=50):
        self._initial_rate = _initial_rate
        self._minimum_rate = _minimum_rate
        self._num_steps = _num_steps

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


class Metric(Callback):
    def __init__(self, model, callbacks, data):
        super().__init__()
        self.model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        X_valid, y_valid = self.data[0], self.data[1]

        y_pred = self.model.predict(X_valid)
        y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (80 * X_valid.shape[0])
        logs['val_CRPS'] = val_s

        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)


def create_model():
    input_dense_players = Input(shape=(11, 10, 10), name='numerical_players_feature_input')
    # Convolutional layers to detect interactions between offense, defense, and the rusher
    x = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), activation='relu')(input_dense_players)
    x = Conv2D(160, kernel_size=(1, 1), strides=(1, 1), activation='relu')(x)
    x = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), activation='relu')(x)
    # Max pooling will find things like the closest/furthest away player from each defender
    # whereas average pooling will find roughly where defenders are in relation to everyone else.
    x_max = MaxPooling2D(pool_size=(1, 10))(x)
    x_avg = AvgPool2D(pool_size=(1, 10))(x)
    # Bias mostly towards the average, but give some weight to the extremes.
    x_max = Lambda(lambda x1: x1 * 0.3)(x_max)
    x_avg = Lambda(lambda x1: x1 * 0.7)(x_avg)
    x = Add()([x_max, x_avg])
    # Squeeze it down to one tensor for each defender.
    x = Lambda(lambda y: tf.squeeze(y, 2))(x)
    x = BatchNormalization()(x)
    x = Conv1D(160, kernel_size=1, strides=1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(96, kernel_size=1, strides=1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(96, kernel_size=1, strides=1, activation='relu')(x)
    x = BatchNormalization()(x)
    # Again, look for a mix of extremes and the averages w.r.t. each defender.
    x_max = MaxPooling1D(pool_size=11)(x)
    x_max = Lambda(lambda x1: x1 * 0.3)(x_max)
    x_avg = AvgPool1D(pool_size=11)(x)
    x_avg = Lambda(lambda x1: x1 * 0.7)(x_avg)
    x = Add()([x_max, x_avg])
    x = Lambda(lambda y: tf.squeeze(y, 1))(x)
    # The final layers of dense outputs
    x = Dense(96, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    per_yard_prob = Dense(80, activation='softmax', name='per_yard_prob')(x)
    model = Model(inputs=[input_dense_players], outputs=per_yard_prob)
    model.compile(loss=CrpsLoss(), optimizer=Adam(learning_rate=MyLearningRateSchedule()))
    return model


def log_validation_predictions(fold_idx, model, x, y_true):
    y_pred = model.predict(x)
    yards = np.arange(-29, 51)
    if fold_idx is None:
        outfilename = 'ery.losses.txt'
    else:
        outfilename = 'ery.losses.fold=%d.txt' % fold_idx
    with open(outfilename, 'w') as outfile:
        for idx in range(len(y_pred)):
            expected_yards = np.sum(y_pred[idx] * yards)
            actual_yards = np.sum(y_true[idx] * yards)
            diff = expected_yards - actual_yards
            print('%04d Expected: %5.2f Actual: %5.2f Diff: %5.2f' % (idx, expected_yards, actual_yards, diff),
                  file=outfile)
    if fold_idx is None:
        model.save('ery.full.%f.model' % (time()))
    else:
        model.save('ery.fold=%d.%f.model' % (fold_idx, time()))


def expand_indexes(fold_indexes, samples_per_play):
    indexes = fold_indexes * samples_per_play
    return np.append(indexes, [indexes + i for i in range(1, samples_per_play)])


def train_one_model(fold_idx, X_train, y_train, x_valid, y_valid, season_valid):
    non_2017_idx = np.where(season_valid != 2017)
    x_valid = x_valid[non_2017_idx]
    y_valid = y_valid[non_2017_idx]
    print('\tTrain size:', len(X_train), len(y_train))
    print('\tValid size:', len(x_valid), len(y_valid), len(season_valid) - len(non_2017_idx))
    model = create_model()
    early_stopping = EarlyStopping(monitor='val_CRPS', mode='min', patience=25, min_delta=1e-3,
                                   restore_best_weights=True)
    early_stopping.set_model(model)
    metric = Metric(model, [early_stopping], [x_valid, y_valid])
    model.fit(X_train,
              y_train,
              epochs=60,
              batch_size=64,
              verbose=1,
              callbacks=[metric],
              validation_data=(x_valid, y_valid))
    val_crps_score = min(model.history.history['val_CRPS'])
    print("\tVal loss: {}".format(val_crps_score))
    log_validation_predictions(fold_idx, model, x_valid, y_valid)
    return model, val_crps_score


def run_cross_fold_training(X_train, y_train, season_train, samples_per_play=3):
    models = []
    scores = []
    assert len(X_train) % samples_per_play == 0, 'Number of samples [%d] is not a multiple of %d' % (
        len(X_train), samples_per_play)
    # While it's okay for the original version of a play to be in the training set and the mirrored version of the
    # play to be in the validation set, we try to keep the same play out of both the training and validation set. The
    # same play can show up multiple times because the order in which each defender and offensive player shows up in
    # the input matrix is shuffled. This gives convolution a better picture but we try to keep all permutations of one
    # play in EITHER the training OR the validation set, but not span both.
    actual_plays = len(X_train) / samples_per_play
    play_index = np.arange(actual_plays)
    k_fold = KFold(n_splits=6, shuffle=True)
    for fold_idx, (raw_train_idx, raw_valid_idx) in enumerate(k_fold.split(play_index)):
        print('EryFold:', fold_idx)
        train_idx = expand_indexes(raw_train_idx, samples_per_play)
        valid_idx = expand_indexes(raw_valid_idx, samples_per_play)
        fold_x_train, fold_x_valid = X_train[train_idx], X_train[valid_idx]
        fold_y_train, fold_y_valid = y_train[train_idx], y_train[valid_idx]
        fold_season_valid = season_train[valid_idx]
        model, val_score = train_one_model(fold_idx, fold_x_train, fold_y_train, fold_x_valid, fold_y_valid,
                                           fold_season_valid)
        scores.append(val_score)
        models.append(model)
    print(np.mean(scores))
