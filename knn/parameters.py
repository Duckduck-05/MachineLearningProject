import logging

import torch
import numpy as np
from matplotlib import pyplot as plt
from utils.helper import standardize, min_max_scale, random_split, sequential_split

plt.style.use("ggplot")
plt.rcParams["font.family"] = "Roboto"
plt.rcParams["axes.labelweight"] = "ultralight"

np.seterr(all="ignore")

logger = logging.getLogger("Parameters")
# The device to train the model on
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using {device}")
# Data's parameters
code = "^GSPC"
start_day = "2020-01-01"
end_day = "2025-01-01"
cols = ["Close"]
seq_length = 180
split_ratio = [0.7, 0.3]
profit_rate = 0.03
use_median = True
prediction_step = 30

transform = standardize
split_func = sequential_split

# K-NN params
k = 1000

# Autoencoder params
latent_size = 12

# Other params
batch_size = 300
wknn_train_split_ratio = 0.99