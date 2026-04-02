from datasets import load_dataset
import os
from config import SEQ_LEN, DATA_DIRNAME, N_VOCAB
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

path_to_shards = os.path.join(DATA_DIRNAME, "train", "*.arrow")
ds = load_dataset("arrow", data_files=path_to_shards, split="train", streaming=True)

tokenizer = Tokenizer(models.BPE())

trainer = trainers.BpeTrainer(
    vocab_size=N_VOCAB,
    min_frequency=2,
    special_tokens=["<pad>", "<bos>", "<eos>"]
)

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()

def batch_iterator(batch_size=1000):
    batch = []
    for example in ds:
        batch.append(example["text"])  # adjust if needed
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

tokenizer.train_from_iterator(batch_iterator(), trainer)

tokenizer.post_processor = processors.TemplateProcessing(
    single="<bos> $A <eos>",
    special_tokens=[
        ("<bos>", tokenizer.token_to_id("<bos>")),
        ("<eos>", tokenizer.token_to_id("<eos>")),
    ],
)

tokenizer.save("tokenizer.json")