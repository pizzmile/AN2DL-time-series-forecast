import tensorflow as tf
import logging

# Setup logger
import yaml

logger = logging.getLogger(__name__)

APPLICATIONS_DICT = {
    'VGG16': tf.keras.applications.vgg16.VGG16,
    'InceptionV3': tf.keras.applications.InceptionV3,
    'ResNet50': tf.keras.applications.resnet50.ResNet50
}

LAYERS_DICT = {
    'Input': tf.keras.layers.Input,
    'Conv2D': tf.keras.layers.Conv2D,
    'MaxPooling2D': tf.keras.layers.MaxPooling2D,
    'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
    'AveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
    'Flatten': tf.keras.layers.Flatten,
    'Dense': tf.keras.layers.Dense,
    'Dropout': tf.keras.layers.Dropout
}

KERNEL_INITIALIZERS_DICT = {
    'GlorotUniform': tf.keras.initializers.GlorotUniform,
}

LOSSES_DICT = {
    'CategoricalCrossentropy': tf.keras.losses.CategoricalCrossentropy
}

OPTIMIZERS_DICT = {
    'Adam': tf.keras.optimizers.Adam,
    'Sigmoid': tf.keras.optimizers.SGD
}


def _parse_mapping(loader: yaml.Loader, node: yaml.Node):
    m = loader.construct_mapping(node, deep=True)
    params = m['params'] if m['params'] is not None else {}
    # logger.debug(f'Parsed map: {m}')
    return m['type'], params


def _tune_network(t: str, p: dict) -> tf.keras.applications.Application:
    # Clean kwargs
    params = p.copy()
    locked_layers = params.pop('locked_layers', None)
    # Set trainable layers
    super_net = APPLICATIONS_DICT[t](**params)
    for i, sub_layer in enumerate(super_net.layers):
        sub_layer.trainable = True if i > locked_layers else False
    return super_net


def build_network(loader: yaml.Loader, node: yaml.Node) -> tf.keras.applications.Application:
    """
    Build a Tensorflow pre-trained network
    :param loader: file loader
    :param node: node to construct
    :return: pre-trained network
    :rtype: tf.keras.applications.Application
    """
    t, p = _parse_mapping(loader, node)
    try:
        network = _tune_network(t, p)
        logger.debug(f'Built loss: {network}')
        return network
    except KeyError:
        logger.warning('Undefined loss exception', exc_info=True)


def build_loss(loader: yaml.Loader, node: yaml.Node) -> tf.keras.losses.Loss:
    """
    Build a Tensorflow loss function
    :param loader: file loader
    :param node: node to construct
    :return: loss function
    :rtype: tf.keras.losses.Loss
    """
    t, p = _parse_mapping(loader, node)
    try:
        loss = LOSSES_DICT[t](**p)
        logger.debug(f'Built loss: {loss}')
        return loss
    except KeyError:
        logger.warning('Undefined loss exception', exc_info=True)


def build_optimizer(loader: yaml.Loader, node: yaml.Node) -> tf.keras.optimizers.Optimizer:
    """
    Build a Tensorflow optimizer function
    :param loader: file loader
    :param node: node to construct
    :return: optimizer function
    :rtype: tf.keras.optimizers.Optimizer
    """
    t, p = _parse_mapping(loader, node)
    try:
        optimizer = OPTIMIZERS_DICT[t](**p)
        logger.debug(f'Built optimizer: {optimizer}')
        return optimizer
    except KeyError:
        logger.warning('Undefined optimizer exception', exc_info=True)


def build_sequential(loader: yaml.Loader, node: yaml.Node) -> tf.keras.Sequential:
    """
    Build a Tensorflow sequential network given its layers
    :param loader: file loader
    :param node: node to construct
    :return: sequential network
    :rtype: tf.keras.Sequential
    """
    s = loader.construct_sequence(node, deep=True)
    # logger.debug(f'Parsed sequence: {s}')
    sequential = tf.keras.Sequential(
        layers=s
    )
    logger.debug(f'Built sequential: {sequential}')
    return sequential


def build_layer(loader: yaml.Loader, node: yaml.Node) -> tf.keras.layers.Layer:
    """
    Build a Tensorflow layer
    :param loader: file loader
    :param node: node to construct
    :return: layer
    :rtype: tf.keras.layers.Layer
    """
    t, p = _parse_mapping(loader, node)
    try:
        layer = LAYERS_DICT[t](**p)
        logger.debug(f'Built layer: {layer}')
        return layer
    except KeyError:
        logger.warning('Undefined layer exception', exc_info=True)


def build_initializer(loader: yaml.Loader, node: yaml.Node) -> tf.keras.initializers.Initializer:
    """
    Build a Tensorflow initializer function
    :param loader: file loader
    :param node: node to construct
    :return: initializer function
    :rtype: tf.keras.initializers.Initializer
    """
    t, p = _parse_mapping(loader, node)
    try:
        initializer = KERNEL_INITIALIZERS_DICT[t](**p)
        logger.debug(f'Built initializer: {initializer}')
        return initializer
    except KeyError:
        logger.warning('Undefined initializer exception', exc_info=True)
