import logging
import logging.config

CONFIG_DIR = "./logging/"


def setup_logging(level='prod'):
    """Load logging configuration"""
    log_configs = {
        'debug': 'logging.dev.ini',
        'info': 'logging.info.ini',
        'prod': 'logging.warn.ini'}
    config = log_configs[level]
    config_path = CONFIG_DIR + config

    logging.config.fileConfig(
        config_path,
        disable_existing_loggers=False)
