from typing import List

import numpy as np
import pandas as pd
from pandas_datareader import data as web

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
    def __init(self, lag:int = 1):
        self.lag = lag

    def transform(self, data: pd.DataFrame):
        # 1-D 'Close' values
        # [179.3, 176.9, 180.1, ...]
        dataset = data['Close'].values.astype('float32')
        dataset = np.reshape(dataset, len(dataset))

        dataset_examples = timeseries_to_supervised(pd.DataFrame(dataset_diffs), sequence_length)

        diffed = data[['Close']].diff(self.lag).dropna()


        return dataset_examples.as_matrix()
