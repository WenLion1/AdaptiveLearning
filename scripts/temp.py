import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from mne.stats import permutation_cluster_1samp_test
from scipy.spatial.distance import squareform
from scipy.stats import vonmises, zscore, spearmanr
from autoreject import AutoReject
import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

if __name__ == "__main__":
    data = np.load("../data/eeg/hc/2base_-1_0.5_baseline(6)_0_0.2/autoreject/npy/405_clean.npy")
    print(data.shape)

    data = np.load("../hidden/sub/hc/405/remove/rnn_layers_1_hidden_16_input_489_combine_processed.npy")
    print(data.shape)