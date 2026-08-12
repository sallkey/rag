import logging
from logging.config import dictConfig


def logger():
    logging_config = {
        'version': 1,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'standard'
            },
            'file': {
                'level': 'WARNING',
                'class': 'logging.FileHandler',
                'filename': 'demo.log',
                'formatter': 'standard'
            }
        },
        'loggers': {
            'logger': {
                'level': 'DEBUG',
                'handlers': ['console'],
                'propagate': False
            }
        }
    }

    dictConfig(logging_config)
    return logging.getLogger('logger')
