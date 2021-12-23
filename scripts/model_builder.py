import os
import pprint
import sys
import tensorflow as tf
import yaml
from scripts import to_float, to_tuple
from scripts import build_layer, build_initializer, build_sequential, build_loss, build_optimizer, build_network
import logging
import numpy as np
import pandas as pd


def dummy_constructor(loader, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    elif isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node, deep=True)
    elif isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return node


# Setup project logger
logger = logging.getLogger(__name__)

# Setup yaml constructors
yaml.add_constructor('!network', build_network)
yaml.add_constructor('!layer', build_layer)
yaml.add_constructor('!initializer', build_initializer)
yaml.add_constructor('!sequential', build_sequential)
yaml.add_constructor('!loss', build_loss)
yaml.add_constructor('!optimizer', build_optimizer)
yaml.add_constructor('!tuple', to_tuple)
yaml.add_constructor('!float', to_float)


class ModelConfig:
    __name: str = None
    __compile_params: dict = {}
    __fit_params: dict = {}
    __input_layer: tf.keras.layers.Layer = {}
    __output_layer: tf.keras.layers.Layer = {}
    __hidden_layers: list = []
    __dataset_params: dict = {}

    def __init__(self,
                 dataset_params: dict, compile_params: dict, fit_params: dict,
                 input_layer: tf.keras.layers.Layer, output_layer: tf.keras.layers.Layer, hidden_layers: list,
                 name: str = None):
        self.__name = name
        self.__compile_params = compile_params
        self.__fit_params = fit_params
        self.__input_layer = input_layer
        self.__output_layer = output_layer
        self.__hidden_layers = hidden_layers
        self.__dataset_params = dataset_params

    @staticmethod
    def load_config(filepath: str):
        """
        Create a ModelConfig object from a configuration .yaml file
        :param filepath: filepath of a .yaml file
        :return: configuration object
        :rtype: ModelConfig
        """
        with open(filepath, 'r') as config_file:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                return ModelConfig(**config)
            except yaml.YAMLError as e:
                print(e)
            finally:
                config_file.close()

    def get_input_layer(self) -> tf.keras.layers.Layer:
        return self.__input_layer

    def get_output_layer(self) -> tf.keras.layers.Layer:
        return self.__output_layer

    def get_hidden_layers(self) -> list:
        return self.__hidden_layers

    def get_name(self) -> str:
        return self.__name

    def get_compile_params(self) -> dict:
        return self.__compile_params

    def get_fit_params(self) -> dict:
        return self.__fit_params

    def get_dataset_params(self) -> dict:
        return self.__dataset_params

    def build_model(self) -> tf.keras.Model:
        """
        Build a model given its configuration
        :return: network model
        :rtype: tf.keras.Model
        """
        input_layer = self.__input_layer
        output_layer = self.__output_layer
        hidden_layers = self.__hidden_layers

        x = input_layer
        for elem in hidden_layers:
            x = elem(x)
        output_layer = output_layer(x)

        m = tf.keras.Model(inputs=input_layer, outputs=output_layer, name=self.__name)
        m.compile(loss=self.__compile_params['loss'],
                  optimizer=self.__compile_params['optimizer'],
                  metrics=self.__compile_params['metrics'])

        return m

    def __split_data(self, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """
        Split a given dataset in train a test partitions, given the test size percentage, and normalize them
        :param df: the dataframe to split
        :return: the train dataframe and the test dataframe
        :rtype: tuple[pandas.DataFrame, pandas.Dataframe]
        """
        # Split dataset
        num_of_records = int(df.shape[0] * self.__dataset_params['test_size'])
        x_train_raw = df.iloc[:-num_of_records]
        x_test_raw = df.iloc[-num_of_records:]
        # Normalize features and labels
        x_min = x_train_raw.min()
        x_max = x_train_raw.max()
        x_train = (x_train_raw - x_min) / (x_max - x_min)
        x_test = (x_test_raw - x_min) / (x_max - x_min)
        return x_train, x_test

    def __build_sequences(self, df: pd.DataFrame, target_labels: list[str]) -> (np.ndarray, np.ndarray):
        """
        Split a dataframe into time series
        :param df: the dataframe to split
        :param target_labels: the list of labels to consider
        :return: the lists of time series and their labels
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        window = self.__dataset_params['window']
        stride = self.__dataset_params['stride']
        telescope = self.__dataset_params['telescope']

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

    def preprocess_data(self, df: pd.DataFrame, target_labels: list[str]
                        ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Preprocess dataset
        :param df: raw dataset
        :param target_labels: labels to keep
        :return: train and test sets
        :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        """
        x_train, x_test = self.__split_data(df)
        x_train, y_train = self.__build_sequences(x_train, target_labels=target_labels)
        x_test, y_test = self.__build_sequences(x_test, target_labels=target_labels)
        return x_train, y_train, x_test, y_test
