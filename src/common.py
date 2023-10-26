import numpy as np
import pandas as pd
import tensorflow as tf


def limit_gpu_memory_growth():
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except RuntimeError as e:
        # Invalid device or cannot modify virtual devices once initialized.
        print(e)


def setup_environment():
    limit_gpu_memory_growth()
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 6)
    np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: '%.4g' % x))
