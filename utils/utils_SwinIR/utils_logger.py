import sys
import datetime
import logging


'''
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# 03/Mar/2019
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


'''
# --------------------------------------------
# logger
# --------------------------------------------
'''


def logger_info(logger_name, log_path='default_logger.log'):
    log = logging.getLogger(logger_name)

    # --- Force removal of any old handlers ---
    if log.hasHandlers():
        for handler in list(log.handlers):
            log.removeHandler(handler)

    # Now set up fresh
    level = logging.INFO
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_path, mode='a')
    fh.setFormatter(formatter)
    log.setLevel(level)
    log.addHandler(fh)

    # Add console stream handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    log.addHandler(sh)



'''
# --------------------------------------------
# print to file and std_out simultaneously
# --------------------------------------------
'''


class logger_print(object):
    def __init__(self, log_path="default.log"):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  # write the message

    def flush(self):
        pass
