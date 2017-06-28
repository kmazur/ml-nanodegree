import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def timeseries_to_supervised(data: pd.DataFrame, sequence_length: int = 1) -> pd.DataFrame:
    """
    | Creates an array of shifted data (as columns)
    |
    | | shift=2 | shift=1 | shift=0 | ...
    | |---------|---------|---------|----
    | |    na   |    na   |    1    | ...
    | |    na   |    1    |    2    | ...
    | |    1    |    2    |    3    | ...
    |
    | and then removes 'na' values
    | and then sets the column names as indices (0, 1, 2, ...)
    | which results in:
    | |    0    |    1    |    2    | ...
    | |---------|---------|---------|----
    | |    1    |    2    |    3    | ...
    | |    2    |    3    |    4    | ...
    | |    3    |    4    |    5    | ...
    |
    :param data: pandas DataFrame
    :param sequence_length: length of the sequence
    :return: pandas DataFrame with rows as sequences of shifted data
    """
    columns = [data.shift(i) for i in range(sequence_length, 0, -1)]
    columns.append(data)
    # Concats arrays to pandas DataFrame and removes n=window_lag first rows
    df2 = pd.concat(columns, axis=1)  # type: pd.DataFrame
    df2.dropna(inplace=True)
    df2.columns = range(sequence_length + 1)
    return df2

def dataframe_to_supervised(data: pd.DataFrame, sequence_length: int = 1) -> list:
    assert sequence_length >= 1
    sequence_length = sequence_length - 1

    shiftFrames = [data.shift(i) for i in range(sequence_length, 0, -1)]
    shiftFrames.append(data)
    for i in range(len(shiftFrames)):
        frame = shiftFrames[i]
        frame.drop(frame.index[:sequence_length], inplace=True)
        # don't need to
    return shiftFrames


def differences_to_original(differences: np.ndarray, initial_values: np.ndarray, diff_lag: int = 1) -> np.ndarray:
    """
    | Example:
    |
    | train_diff = pd.DataFrame(train).diff(lag).dropna()
    | diff_series = train_diff[0].values  # type: np.ndarray
    | reversed = differences_to_original(diff_series, train[0:lag], lag)
    |

    :param differences: array of differences e.g. after pd.DataFrame.diff(diff_lag)
    :param initial_values: the first values in the original series
    :param diff_lag: how many periods in the differences was used
    :return: original series
    """
    reverted = np.concatenate((initial_values, differences))
    for i in range(len(differences)):
        reverted[i + diff_lag] = reverted[i] + reverted[i + diff_lag]
    return reverted


def normalise_windows(window_data: np.ndarray) -> np.ndarray:
    """
    | Example:
    | from: [[2, 4, 6], [2, 3, 4]]
    | to:   [[0, 1, 2], [0, 0.5, 1]]
    |

    :param window_data: [[1, 2, 3, ...], [2, 3, 4, ...]] array of sequences
    :return: transformed windows as a relative value to first window element (w_n/w_0)
    """
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return np.asarray(normalised_data)


def diff(series: np.ndarray, lag=1) -> np.ndarray:
    """
    :param series: 1-D array
    :param lag: diff lag
    :return: 1-D array of diffs
    """
    values = pd.DataFrame(series).diff(lag).dropna().values
    return np.reshape(values, len(values))

def to_examples(dataset: np.ndarray, diff_lag: int = 1, sequence_length: int = 50) -> np.ndarray:
    dataset_diffs = diff(dataset, lag=diff_lag)
    dataset_examples = timeseries_to_supervised(pd.DataFrame(dataset_diffs), sequence_length)
    return dataset_examples.as_matrix()

def from_predictions(predictions: list, examples: np.ndarray, original: np.ndarray, diff_lag: int, prediction_length: int) -> list:
    real_data = []
    for i in range(len(predictions)):
        prediction = predictions[i]                                             # [3]           -> diff prediction
        example = examples[i * prediction_length]                               # [1, -2]       -> diff example
        init = np.array([original[i * prediction_length]])                      # [1]           -> first value in example before diff

        # [1, -2, 3]
        diff_series = np.concatenate((example, prediction))
        # [1] + [1, -2, 3] => [1, 2, 0, 3]
        real_example_with_predictions = differences_to_original(diff_series,
                                                                initial_values=init,
                                                                diff_lag=diff_lag)

        # [3]
        real_predictions = real_example_with_predictions[len(init) + len(example):]
        # [..., [3]]
        real_data.append(real_predictions.tolist())

    return real_data


def split_train_test(dataset_examples, ratio: float = 0.8):
    train_size = int(len(dataset_examples) * ratio)
    train = dataset_examples[0:train_size]  # type: np.ndarray
    test = dataset_examples[train_size:]    # type: np.ndarray

    X_train = train[:, 0:-1]
    X_test = test[:, 0:-1]
    y_train = train[:, -1]
    y_test = test[:, -1]
    return X_train, X_test, y_train, y_test