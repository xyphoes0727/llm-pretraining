from typing import Tuple
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch
from config import BATCH_SIZE
torch.manual_seed(42)


def load_dataset(data_path: str) -> Tuple[DataLoader, DataLoader]:
    dataset = load_from_disk(data_path)

    train_dataset = dataset["train"].with_format("torch")
    test_dataset = dataset["test"].with_format("torch")

    def collate_fn(batch):
        batched_seqs = [seq["input_ids"] for seq in batch]
        padded_seqs = pad_sequence(
            batched_seqs, batch_first=True, padding_value=0)
        return padded_seqs

    train_data_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, pin_memory=True)
    test_data_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, pin_memory=True)
    return (train_data_loader, test_data_loader)
