import logging
import torch
import numpy as np
from matplotlib import pyplot as plt
from utils.helper import standardize, min_max_scale, sequential_split

plt.style.use("ggplot")
plt.rcParams["font.family"] = "Roboto"
plt.rcParams["axes.labelweight"] = "ultralight"

np.seterr(all="ignore")

logger = logging.getLogger("Parameters")
# The device to train the model on
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using {device}")
# Data's parameters
code = "^GSPC"  # Symbol for S&P 500
# Adjust date format to match dataset (MM/DD/YYYY)
start_day = "01/01/2020"  # Start date within dataset range
end_day = "04/30/2025"    # End date within dataset range (latest available)
cols = ["Close"]  # Use 'Close' column from sp500.csv
seq_length = 10
split_ratio = [0.60, 0.20, 0.20]  # Train, validation, test split
profit_rate = 0.03  # 3% profit threshold
use_median = True
prediction_step = 1

transform = standardize  # Use standardization as the default transform
split_func = sequential_split  # Use sequential split for time-series data

# K-NN params
k = 6  # Number of neighbors, reasonable for small sequences

# Autoencoder params
latent_size = 5  # Dimension of latent space, suitable for 1D close price data

# Other params
batch_size = 64  # Batch size for training, suitable for most datasets
wknn_train_split_ratio = 0.8  # Ratio for weighted KNN training split
learning_rate = 0.05  # Learning rate for model training