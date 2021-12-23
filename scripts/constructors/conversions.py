import logging

import yaml


logger = logging.getLogger(__name__)


def to_tuple(loader: yaml.Loader, node: yaml.Node) -> tuple:
    """
    Convert a Yaml list into a Python tuple
    :param loader: file loader
    :param node: node to construct
    :return: converted tuple
    :rtype: tuple
    """
    seq = loader.construct_sequence(node)
    logger.debug('Yaml ("to_tuple") -> %s', str(seq))
    return tuple(seq)


def to_float(loader: yaml.Loader, node: yaml.Node) -> float:
    """
    Convert the string scientific notation of a float number into its Python form
    :param loader: file loader
    :param node: node to construct
    :return: converted float
    :rtype: float
    """
    scalar = loader.construct_scalar(node)
    logger.debug('Yaml ("to_float") -> %s', str(scalar))
    return float(scalar)


def map_to_list(loader: yaml.Loader, node: yaml.Node) -> list:
    """
    Transform a PyYaml map to a list
    :param loader: file loader
    :param node: node to construct
    :return: converted list
    :rtype: list
    """
    mapping = loader.construct_mapping(node)
    seq = [mapping[key] for key in mapping.keys()]
    logger.debug('Yaml ("map_to_list") -> %s', str(seq))
    return seq
