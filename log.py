import logging

__loggers = []


def getLogger(name):
    logger = logging.getLogger(name)
    console = logging.StreamHandler()
    fmt = logging.Formatter('[%(levelname)s] (%(asctime)s) %(message)s')
    console.setFormatter(fmt)
    logger.addHandler(console)
    __loggers.append(logger)
    return logger

def setLevel(level):
    for logger in __loggers:
        logger.setLevel(level)
