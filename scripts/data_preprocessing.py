import pandas as pd
import numpy as np


def split_data(df: pd.DataFrame, test_size: float) -> (pd.DataFrame, pd.DataFrame):
    """
    Split a given dataset in train a test partitions, given the test size percentage
    :param df: the dataframe to split
    :param test_size: the percentage of the test partition over the entire dataframe
    :return: the train dataframe and the test dataframe
    :rtype: tuple[pandas.DataFrame, pandas.Dataframe]
    """
    # Split dataset
    num_of_records = int(df.shape[0] * test_size)
    x_train = df.iloc[:-num_of_records]
    x_test = df.iloc[-num_of_records:]
    return x_train, x_test


def normalize_data(x_train_raw: pd.DataFrame, x_test_raw: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Normalize the data of train and test dataframes
    :param x_train_raw: raw train dataframe
    :param x_test_raw: raw test dataframe
    :return: the normalized train and test dataframes
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    """
    # Normalize features and labels
    x_min = x_train_raw.min()
    x_max = x_train_raw.max()
    x_train = (x_train_raw - x_min) / (x_max - x_min)
    x_test = (x_test_raw - x_min) / (x_max - x_min)
    return x_train, x_test


def build_sequences(df: pd.DataFrame, target_labels: list[str],
                    window: int, stride: int, telescope: int) -> (np.ndarray, np.ndarray):
    """
    Split a dataframe into time series
    :param df: the dataframe to split
    :param target_labels: the list of labels to consider
    :param window: the number of outputs per sequence
    :param stride: the distance of the inputs form outputs
    :param telescope: the number of inputs for each sequence
    :return: the lists of time series and their labels
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    # Sanity check
    assert window % stride == 0

    dataset = []
    labels = []
    temp_df = df.copy().values
    temp_labels = df[target_labels].copy().values
    padding_len = len(df) % window

    if padding_len != 0:
        # Compute padding length
        padding_len = window - padding_len
        padding = np.zeros((padding_len, temp_df.shape[1]), dtype='float64')
        temp_df = np.concatenate((padding, df))
        padding = np.zeros((padding_len, temp_labels.shape[1]), dtype='float64')
        temp_labels = np.concatenate((padding, temp_labels))
        assert len(temp_df) % window == 0

    for idx in np.arange(0, len(temp_df) - window - telescope, stride):
        dataset.append(temp_df[idx:idx + window])
        labels.append(temp_labels[idx + window:idx + window + telescope])

    dataset = np.array(dataset)
    labels = np.array(labels)
    return dataset, labels
