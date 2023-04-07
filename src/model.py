import logging
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys
import tensorflow as tf
import xgboost as xgb

from matplotlib import pyplot
from plays import load_plays_data
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from standardize import normalize_column_formatting
from tensorflow.keras import layers
from tensorflow import keras
from time import time


def plot_xgboost(model):
    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    # plot classification error
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
    ax.legend()
    pyplot.ylabel('Regression Error')
    pyplot.title('XGBoost Regression Error')
    pyplot.show()


def xgboost_model(x_df, y_df, validation_df):
    X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=47)
    # We choose our parameters to avoid overfitting the data. Namely:
    # - Don't make trees too deep, since the number of features we have is relatively small;
    # - Throttle back the learning rate to avoid overfitting;
    # - Only use a sample of the data each time; and
    # - Add an L2 regularization term to avoid selecting too many features.
    model = xgb.XGBRegressor(eval_metric='rmse',
                             booster='dart',
                             max_depth=4,
                             eta=0.025,
                             n_estimators=100,
                             subsample=0.6,
                             reg_lambda=2,
                             verbosity=1,
                             )
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    # make predictions for test data
    y_pred = model.predict(X_test)
    print('XgbMSE: %.6f' % mean_squared_error(y_test, y_pred))
    validation_y_df = validation_df['space']
    validation_x_df = validation_df.drop(columns=['space'])
    v_predictions = model.predict(validation_x_df)
    print('XgbValMSE: %.6f' % mean_squared_error(validation_y_df, v_predictions))
    # plot_xgboost(model)


def stats_model(x_df, y_df, validation_df):
    X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=47)
    X_train = X_train.copy()
    X_train['space'] = y_train
    description = 'space ~ %s' % ' +  '.join([c for c in X_train.columns if c != 'space'])
    # print(description)
    model = smf.ols(description, X_train).fit()
    y_pred = model.predict(X_test)
    print('OlsMSE: %.6f' % mean_squared_error(y_test, y_pred))
    validation_y_df = validation_df['space']
    validation_x_df = validation_df.drop(columns=['space'])
    v_pred = model.predict(validation_x_df)
    print('OlsValMSE: %.6f' % mean_squared_error(validation_y_df, v_pred))
    # print(model.summary())
    table = sm.stats.anova_lm(model, typ=2)
    # print(table)
    sig = list()
    for row in table.iterrows():
        if np.isnan(row[1]['F']):
            continue
        if row[1]['F'] > model.fvalue:
            sig.append(str(row[0]))
    if sig:
        print('\tSignificant columns: [ %s ]' % ', '.join(sig))
    else:
        print('\t*** No significant columns')


def make_keras_model(X_train):
    model = keras.Sequential()
    model.add(
        layers.Dense(input_shape=(X_train.shape[1:]), use_bias=True, units=X_train.shape[1], activation='sigmoid')
    )
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(units=192, activation='sigmoid'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(units=32, activation='sigmoid'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(units=1))
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    model.build()
    print(model.summary())
    return model


def keras_model(x_df, y_df, validation_df):
    X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=47)
    model = make_keras_model(X_train)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, min_delta=1)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=200, shuffle=True, verbose=0,
              callbacks=[early_stopping])
    losses = model.evaluate(x=X_test, y=y_test)
    print('KerMSE: [%s]' % ', '.join(['%.6f' % loss for loss in losses]))
    validation_y_df = validation_df['space']
    validation_x_df = validation_df.drop(columns=['space'])
    v_pred = model.predict(validation_x_df)
    print('KerValMSE: %.6f' % mean_squared_error(validation_y_df, v_pred))
    model.save('data/2019/df/keras.%f.model' % time())


def create_onehot_encoder(plays_df):
    encoder = LabelBinarizer()
    teams = plays_df['possessionTeam'].to_numpy().reshape(-1, 1)
    encoder.fit(teams)
    return encoder


def include_possession_team_abbr(plays_df, encoder, df):
    if 'gameId' not in plays_df or 'playId' not in plays_df:
        return df
    if 'gameId' not in df or 'playId' not in df:
        return df
    plays_df = plays_df[['gameId', 'playId', 'possessionTeam']]
    df = pd.merge(df, plays_df, on=['gameId', 'playId'], how='left')
    possession_team_array = encoder.transform(df['possessionTeam'])
    columns = ['off_%s' % team for team in encoder.classes_]
    possession_team_df = pd.DataFrame(data=possession_team_array, columns=columns)
    df = pd.merge(df, possession_team_df, left_index=True, right_index=True)
    df.drop(columns=['possessionTeam'], inplace=True)
    return df


def main(argv):
    tf.get_logger().setLevel(logging.WARNING)
    if len(argv) != 5:
        print('Usage: %s <games_csv> <plays_csv> <input_csv> <validation>' % argv[0])
        return 1
    games_df = normalize_column_formatting(pd.read_csv(argv[1]))
    plays_df = load_plays_data(games_df, argv[2])
    encoder = create_onehot_encoder(plays_df)
    x_df = pd.read_csv(argv[3])
    x_df = include_possession_team_abbr(plays_df, encoder, x_df)
    x_df['yardsToGoSqrt'] = x_df['yardsToGo'].apply(math.sqrt)
    y_df = x_df['space']
    x_df.drop(columns=['gameId', 'playId', 'space'], inplace=True)
    validation_df = pd.read_csv(argv[4])
    validation_df = include_possession_team_abbr(plays_df, encoder, validation_df)
    validation_df['yardsToGoSqrt'] = validation_df['yardsToGo'].apply(math.sqrt)
    validation_df.drop(columns=['gameId', 'playId'], inplace=True)
    xgboost_model(x_df, y_df, validation_df)
    stats_model(x_df, y_df, validation_df)
    keras_model(x_df, y_df, validation_df)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
