import numpy as np
import torch
import os
import sys
import random
from PIL import Image
from scipy.special import softmax
import matplotlib.pyplot as plt
from modulation_utils import *

selected_channel_idxes = [56, 50, 9]
fs = 250

target_eeg_path = 'server/pre_eeg/10.npy'

target_psd = load_target_psd(target_eeg_path, fs, selected_channel_idxes)

print(target_psd.shape)
