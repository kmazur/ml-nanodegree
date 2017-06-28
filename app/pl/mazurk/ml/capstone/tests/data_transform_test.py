import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_frame_equal

import app.pl.mazurk.ml.capstone.series as series
from app.pl.mazurk.ml.capstone.data import DataTransformation


class DataTranformTest(unittest.TestCase):
    def test_shift_1d_one_lag_dataframe(self):
        # given
        data = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0])
        sequence_length = 1
        expected = [
            pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0])
        ]

        # when
        result = series.dataframe_to_supervised(data, sequence_length)

        # then
        for i in range(len(result)):
            assert_frame_equal(expected[i], result[i])

    def test_shift_1d_three_lag_dataframe(self):
        # given
        data = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0])
        sequence_length = 3
        expected = [
            pd.DataFrame([1.0, 2.0, 3.0]),
            pd.DataFrame([2.0, 3.0, 4.0]),
            pd.DataFrame([3.0, 4.0, 5.0])
        ]

        # when
        result = series.dataframe_to_supervised(data, sequence_length)

        # then
        for i in range(len(result)):
            assert_frame_equal(expected[i], result[i])

    def test_shift_2d_three_lags_dataframe(self):
        # given
        data = pd.DataFrame({
            0: [1.0, 2.0, 3.0, 4.0, 5.0],
            1: ['a', 'b', 'c', 'd', 'e'],
        })

        sequence_length = 3
        expected = [
            pd.DataFrame({0: [1., 2., 3.], 1: ['a', 'b', 'c']}),
            pd.DataFrame({0: [2., 3., 4.], 1: ['b', 'c', 'd']}),
            pd.DataFrame({0: [3., 4., 5.], 1: ['c', 'd', 'e']})
        ]

        # when
        result = series.dataframe_to_supervised(data, sequence_length)

        # then
        for i in range(len(result)):
            assert_frame_equal(expected[i], result[i])

    def test_tranformation_to_rnn_examples(self):
        # given
        data = [
            pd.DataFrame({0: [1.0, 2.0, 3.0], 1: [-1.0, -2.0, -3.0]}),
            pd.DataFrame({0: [2.0, 3.0, 4.0], 1: [-2.0, -3.0, -4.0]}),
            pd.DataFrame({0: [3.0, 4.0, 5.0], 1: [-3.0, -4.0, -5.0]})
        ]
        expected = np.array([
            [[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]],
            [[2.0, -2.0], [3.0, -3.0], [4.0, -4.0]],
            [[3.0, -3.0], [4.0, -4.0], [5.0, -5.0]],
        ])

        # when
        result = series.supervised_to_rnn_examples(data)

        # then
        assert_array_equal(expected, result)

    def test_tranformator_to_rnn_examples_with_one_feature(self):
        # given
        transformator = DataTransformation(sequence_length=2, diff_lag=1)
        data = pd.DataFrame([1, 2, 0, 3])
        expected = np.array([
            [[1], [-2]],
            [[-2], [3]]
        ])

        # when
        result = transformator.transform(data)  # type: np.ndarray

        # then
        assert_array_equal(expected, result)

    def test_inverse_tranformator_from_rnn_predictions_with_one_feature(self):
        # given
        transformator = DataTransformation(sequence_length=2, diff_lag=1)
        data = pd.DataFrame([1, 2, 0, 3])
        examples = np.array([
            [[1], [-2]],
            [[-2], [3]]
        ])
        predictions = np.array([[3], [1]])
        initial_values = np.array([[1], [2]])
        expected = np.array([
            [[1], [2], [0], [3]],
            [[2], [0], [3], [4]],
        ])

        # when
        result = transformator.inverse_transform(predictions, examples, initial_values)  # type: np.ndarray

        # then
        assert_array_equal(expected, result)
