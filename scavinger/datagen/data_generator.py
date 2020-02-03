# -*- coding: utf-8 -*-

# Python imports

# Other imports
import numpy as np
from tensorflow.keras.utils import Sequence

# This package imports
from ..utils import math

class RMPGenerator(Sequence):
    """
    Random Multi-Peak Generator
    
    Generates a random set of peaks with some gaussian noise added on
    for training LSTMs. Also implements an optional background.
    """
    
    def __init__(self, array_length, kind, npeak_range, mu_range, amp_range, sig_range, alpha_range=None):
        pass