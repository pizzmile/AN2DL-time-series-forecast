#%%
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import os
import yaml
from pprint import pprint

from scripts import *

#%%
# Setup project logger
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
fh = logging.FileHandler(f'{os.path.basename(__file__).split(".")[0]}.log')
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[fh, ch],
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup yaml constructors
yaml.add_constructor('!join', join)
yaml.add_constructor('!joinPath', join_path)
yaml.add_constructor('!tuple', to_tuple)
yaml.add_constructor('!float', to_float)
yaml.add_constructor('!divide', divide)
yaml.add_constructor('!mapToList', map_to_list)


if __name__ == '__main__':
    #%%
    # Load configuration
    with open("config.yaml", 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        logger.debug("Configuration loaded")
        config_file.close()

    #%%
    # Create directories
    createDirectoryTree([config['dirs']['data'][k] for k in config['dirs']['data'].keys()])
    createDirectoryTree([config['dirs']['models'][k] for k in config['dirs']['models'].keys()])

    #%%
    # Load model
    filename = os.path.join(config["dirs"]["models"]["settings"], 'model00.yaml')
    model_config = ModelConfig.load_config(filename)
    model = build_model(model_config)

    #%%
    # Load data

    #%%
    # Train model

    #%%
    # Test model
