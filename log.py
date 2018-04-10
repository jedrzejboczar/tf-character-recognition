import logging

__loggers = []
__level = None

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    # def cprint(self, startCodes, text, **kwargs):
    #     startString = ''.join(startCodes)
    #     endString = self.ENDC * len(startCodes)
    #     print(startString + text + endString, **kwargs)

def getLogger(name):
    global __level

    logger = logging.getLogger(name)
    if __level is not None:
        logger.setLevel(__level)
    console = logging.StreamHandler()
    fmt = logging.Formatter('{}[%(levelname)s] %(name)s (%(asctime)s) : %(message)s{}'.format(
        Colors.BOLD, Colors.ENDC))
    console.setFormatter(fmt)
    logger.addHandler(console)
    __loggers.append(logger)
    return logger

def setLevel(level):
    global __level

    __level = level
    for logger in __loggers:
        logger.setLevel(level)
