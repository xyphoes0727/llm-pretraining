from datasets import load_dataset, Dataset
import os
from config import SEQ_LEN, DATA_DIRNAME
from tokenizers import Tokenizer, pre_tokenizers, decoders

path_to_shards = os.path.join(DATA_DIRNAME, "train", "*.arrow")
dataset = load_dataset("arrow", data_files=path_to_shards, split="train")

tokenizer = Tokenizer.from_file("tokenizer.json")
tokenizer.enable_truncation(max_length=SEQ_LEN)
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()

def tokenize_fn(batch):
    out = tokenizer.encode_batch(batch["text"])
    out_ids = [seq.ids for seq in out]
    return {"input_ids": out_ids}

ds = dataset.map(
    tokenize_fn,
    batched=True,
    num_proc=4,
    remove_columns=dataset.column_names
)

dd = ds.train_test_split(test_size=0.02,seed=42)
dd.save_to_disk(r"./tokenized_data_split")