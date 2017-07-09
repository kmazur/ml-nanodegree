from typing import List

import numpy as np
import pandas as pd
from pandas_datareader import data as web
from operator import itemgetter

from app.pl.mazurk.ml.capstone import series


def get_ticker(name, start, end):
    """
    :param name: ticker name
    :param start: start datetime
    :param end: end datetime
    :return: pandas DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
    """
    return web.DataReader(name, 'google', start, end)


def dataframe_to_ndarray(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """
    :param df: pandas DataFrame
    :param columns: columns to extract
    :return: extracted columns values as float32 numpy array
    """
    return df[columns].values.astype('float32')


class DataTransformation:
    def __init__(self, sequence_length: int, diff_lag: int = 1):
        assert sequence_length > 1
        assert diff_lag >= 1
        self.diff_lag = diff_lag
        self.sequence_length = sequence_length

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        diffed = data.diff(self.diff_lag).dropna()
        supervised = series.dataframe_to_supervised(diffed, self.sequence_length)
        rnn_examples = series.supervised_to_rnn_examples(supervised)  # type: np.ndarray
        return rnn_examples

    def inverse_transform(self, predictions: np.ndarray, examples: np.ndarray, initial_values: np.ndarray):
        # Don't use return_sequences at the last step in LSTM layers
        #
        # prediction =
        # [
        #    [[y0_f0, y0_f1], [y1_f0, y1_f1], ...], -> prediction for 0th example with `n` outputs / features
        #   ...
        # ]
        #
        # (x0,x1 - features; t0,t1 - time steps)
        # examples =
        # [
        #   [ [x0_t0, x1_t0], [x0_t1, x1_t1], ... ], -> 0th example from `transform` method
        #   ...
        # ]
        #
        # initial_values =
        # [
        #   [x0_t0, x1_t0],             -> the 0th example initial value (before diff)
        #   ...
        # ]

        # it is a holder for real/original example + prediction series (last value in example is the prediction)
        predicted_series = []
        for prediction_index in range(len(predictions)):
            # [[y0_f0, y0_f1], [y1_f0, y1_f1], ...] - `n` features predicted in prediction_length sequence
            prediction = predictions[prediction_index]
            # [ [x0_t0, x1_t0], [x0_t1, x1_t1], ... ] - series preceding the prediction
            example = examples[prediction_index]

            series_with_prediction = np.concatenate((example, prediction))
            # decompose to arrays of single features:
            # [ x0_t1, x0_t2, ... ]
            # [ x1_t1, x1_t2, ... ]
            # and calculate the real prediction before diff operation
            shape = np.shape(prediction)
            features_count = itemgetter(1)(shape)

            # real_predictions will be (o - original from example, p - prediction):
            # [
            #   [o0_t0, o0_t1, ..., p0_tn],
            #   [o1_t0, o1_t1, ..., p1_tn],
            # ]
            real_predictions = []
            for feature_index in range(features_count):
                feature_series = series_with_prediction[:, feature_index]
                initial_value = initial_values[prediction_index, feature_index]
                original = series.differences_to_original(feature_series, initial_value, self.diff_lag)
                real_predictions.append(original[self.sequence_length:])

            # join back the decomposed features for this example. It will be (note that this is for one example):
            # [
            #   [o0_t0, o1_t0], ..., [p0_tn, p1_tn]
            # ]
            real_timesteps = []
            for timestep_index in range(len(prediction)):
                # iterate over all timesteps
                features_at_timestep = []
                for feature_index in range(features_count):
                    # iterate over features for this timestep
                    features_at_timestep.append(real_predictions[feature_index][timestep_index])
                real_timesteps.append(features_at_timestep)

            predicted_series.append(real_timesteps)

        return np.array(predicted_series)