import warnings
warnings.simplefilter('error', RuntimeWarning)
import logging
import os
from pipeline import Pipeline
from config import get_config_from_command, get_config_from_file


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    # filename='full.log'
)

if __name__ == '__main__':
    config = get_config_from_file()
    pipeline = Pipeline(config)
    pipeline.boost()
