import os
import pprint
import sys
import tensorflow as tf
import yaml
from scripts import to_float, to_tuple
from scripts import build_layer, build_initializer, build_sequential, build_loss, build_optimizer, build_network
import logging


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

    def __init__(self,
                 compile_params: dict, fit_params: dict,
                 input_layer: tf.keras.layers.Layer, output_layer: tf.keras.layers.Layer,
                 hidden_layers: list,
                 name: str = None):
        self.__name = name
        self.__compile_params = compile_params
        self.__fit_params = fit_params
        self.__input_layer = input_layer
        self.__output_layer = output_layer
        self.__hidden_layers = hidden_layers

    @staticmethod
    def load_config(filepath: str) -> object:
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


def build_model(config: ModelConfig) -> tf.keras.Model:
    """
    Build a model given its configuration
    :param config: model configuration in ModelConfig form
    :return: network model
    :rtype: tf.keras.Model
    """
    input_layer = config.get_input_layer()
    output_layer = config.get_output_layer()
    hidden_layers = config.get_hidden_layers()

    x = input_layer
    for elem in hidden_layers:
        x = elem(x)
    output_layer = output_layer(x)

    m = tf.keras.Model(inputs=input_layer, outputs=output_layer, name=config.get_name())
    m.compile(loss=config.get_compile_params()['loss'],
              optimizer=config.get_compile_params()['optimizer'],
              metrics=config.get_compile_params()['metrics'])

    return m


# if __name__ == '__main__':
#     filename = '../models/settings/model00/model00.yaml'
#     model_config = ModelConfig.load_config(filename)
#     model = build_model(model_config)
