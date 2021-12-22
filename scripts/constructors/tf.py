import tensorflow as tf
import logging

# Setup logger
logger = logging.getLogger(__name__)

LAYERS_DICT = {
    'Input': tf.keras.layers.Input,
    'Conv2D': tf.keras.layers.Conv2D,
    'MaxPooling2D': tf.keras.layers.MaxPooling2D,
    'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
    'AveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
    'Flatten': tf.keras.layers.Flatten,
    'Dense': tf.keras.layers.Dense,
    'Dropout': tf.keras.layers.Dropout,
    'VGG16': tf.keras.applications.vgg16.VGG16,
    'InceptionV3': tf.keras.applications.InceptionV3,
    'ResNet50': tf.keras.applications.resnet50.ResNet50,
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


def parse_mapping(loader, node):
    m = loader.construct_mapping(node, deep=True)
    params = m['params'] if m['params'] is not None else {}
    # logger.debug(f'Parsed map: {m}')
    return m['type'], params


# TODO add network constructor


def build_loss(loader, node) -> tf.keras.losses.Loss:
    t, p = parse_mapping(loader, node)
    try:
        loss = LOSSES_DICT[t](**p)
        logger.debug(f'Built loss: {loss}')
        return loss
    except KeyError:
        logger.warning('Undefined loss exception', exc_info=True)


def build_optimizer(loader, node) -> tf.keras.optimizers.Optimizer:
    t, p = parse_mapping(loader, node)
    try:
        optimizer = OPTIMIZERS_DICT[t](**p)
        logger.debug(f'Built optimizer: {optimizer}')
        return optimizer
    except KeyError:
        logger.warning('Undefined optimizer exception', exc_info=True)


def build_sequential(loader, node) -> tf.keras.Sequential:
    s = loader.construct_sequence(node, deep=True)
    # logger.debug(f'Parsed sequence: {s}')
    sequential = tf.keras.Sequential(
        layers=s
    )
    logger.debug(f'Built sequential: {sequential}')
    return sequential


def build_layer(loader, node) -> tf.keras.layers.Layer:
    t, p = parse_mapping(loader, node)
    try:
        layer = LAYERS_DICT[t](**p)
        logger.debug(f'Built layer: {layer}')
        return layer
    except KeyError:
        logger.warning('Undefined layer exception', exc_info=True)


def build_initializer(loader, node) -> tf.keras.initializers.Initializer:
    t, p = parse_mapping(loader, node)
    try:
        initializer = KERNEL_INITIALIZERS_DICT[t](**p)
        logger.debug(f'Built initializer: {initializer}')
        return initializer
    except KeyError:
        logger.warning('Undefined initializer exception', exc_info=True)
