__author__ = 'lucas'


import numpy as np
import matplotlib.pyplot as plt
import pickle
from plots import ExperimentData



if __name__ == "__main__":
    exp = ExperimentData.unpickle('28.pkl')
    exp.plot_new()