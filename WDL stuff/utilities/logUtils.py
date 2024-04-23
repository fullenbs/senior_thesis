"""
This file contains various utitilities to interface with the saved log files. In the future it may be nice to migrate
 this to a centralized database instead of individual files per run, but I don't want to prematurely optimize
"""

import pickle
import torch
import datetime


def saveLog(log: dict, support: torch.Tensor, directory=""):
    """
    saves thes specified log along with the data's support in a log file

    :param log: the log dictionary to save
    :param support: the support of the distribution
    TODO: add ways of differentiation between image support and 1D support
    :return:
    """
    log["support"] = support
    # name the file after the current timestamp
    fname = directory + "log: " + str(datetime.datetime.now()) + ".pickle"
    with open(fname, 'wb') as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)


def readLogFile(fname: str):
    """
    returns the dictionary saved in the log file

    :param fname: the log filename
    :return: the log dictionary
    """
    with open(fname, 'rb') as handle:
        log = pickle.load(handle)
    return log
