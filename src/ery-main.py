#!python

import erylib
import numpy as np
import sys

from common import setup_environment
from sklearn.model_selection import train_test_split


def create_transformation_matrix():
    t = np.zeros((10, 10))
    for idx in range(0, 5):
        idx_2 = idx * 2
        idx_2_1 = idx * 2 + 1
        t[idx_2][idx_2_1] = -1
        t[idx_2_1][idx_2] = 1
    return t


def main(argv):
    if len(argv) != 2:
        print('Usage: %s <train_csv>' % argv[0])
        return 1
    setup_environment()
    df_train = erylib.load_data(argv[1])
    train_x_ery, train_y, seasons = erylib.create_training_inputs(df_train, samples_per_play=3)
    X_train, x_valid, y_train, y_valid, _, seasons_valid = train_test_split(train_x_ery, train_y, seasons,
                                                                            test_size=0.2)
    erylib.train_one_model(None, X_train, y_train, x_valid, y_valid, seasons_valid)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
