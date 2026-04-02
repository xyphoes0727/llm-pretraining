import torch
from datasets import load_dataset
from torch.utils.data import Dataset
import os
from config import SEQ_LEN, DATA_DIRNAME

data_dir = os.path.join(os.getcwd(), DATA_DIRNAME)
ds = load_dataset("Skylion007/openwebtext")
ds.save_to_disk(data_dir)
