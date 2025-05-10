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
seq_length = 10
split_ratio = [0.60, 0.20, 0.20]
profit_rate = 0.03
use_median = True
prediction_step = 1

transform = standardize
split_func = random_split

# K-NN params
k = 6  # Updated to a more reasonable value

# Autoencoder params
latent_size = 5  # Updated to a more suitable value

# Other params
batch_size = 64  # Updated to a more suitable value
wknn_train_split_ratio = 0.8
learning_rate = 0.05