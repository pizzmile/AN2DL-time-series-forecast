import logging
import os
import yaml

logger = logging.getLogger(__name__)


def join(loader: yaml.Loader, node: yaml.Node) -> str:
    """
    Join the elements of a list into a single string
    :param loader: file loader
    :param node: node to construct
    :return: concatenated string
    :rtype: str
    """
    seq = loader.construct_sequence(node)
    logger.debug('Yaml ("join") -> %s', str(seq))
    return ''.join([str(i) for i in seq])


def join_path(loader: yaml.Loader, node: yaml.Node) -> str:
    """
    Join the elements of a list into a path string through os.path.join
    :param loader: file loader
    :param node: node to construct
    :return: concatenated path string
    :rtype: str
    """
    seq = loader.construct_sequence(node)
    logger.debug('Yaml ("join_path") -> %s', str(seq))
    return os.path.join(*seq)
