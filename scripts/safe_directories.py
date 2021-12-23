import os
import logging

logger = logging.getLogger(__name__)


def create_directory_tree(tree: list[str]) -> int:
    """
    Create a directory tree safely, given the list of its directories
    :param tree: list of directories names
    :return number of directories that have been created
    :rtype: int
    """
    created_directories = 0
    for directory in tree:
        if not os.path.exists(directory):
            os.makedirs(directory)
            created_directories += 1
            logger.info("Created directory {directory}".format(directory=directory))

    return created_directories
