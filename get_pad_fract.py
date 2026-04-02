import time
import logging
import torch
import torch.nn as nn
import config
from torch.nn.utils import clip_grad
from model import MyModel
import logging
from dataset import load_dataset
import wandb
from dotenv import load_dotenv
import os
import math

# Check padding fraction of the dataset for debugging purposes.

train_data_loader, test_data_loader = load_dataset(config.DATASET_PATH)

i = 0
for batch_data in train_data_loader:
    if(i >= 80):
        break
    is_pad = (batch_data == 0).float()

    pad_fracts = torch.mean(is_pad, dim=-1)
    print(pad_fracts)
    break