import unittest
import app.pl.mazurk.ml.capstone.series as series
import pandas as pd
import numpy as np

class DataTranformTst(unittest.TestCase):

    def should_shift_1d_one_lag_dataframe_correctly(self):
        # given
        data = pd.DataFrame([1, 2, 3, 4, 5])
        sequence_length = 3
        expected = [
            pd.DataFrame([1, 2, 3]),
            pd.DataFrame([2, 3, 4]),
            pd.DataFrame([4, 5, 6])
        ]

        # when
        result = series.dataframe_to_supervised(data, sequence_length)

        print(expected)
        print(result)
        # then
        self.assertEqual(expected, result)