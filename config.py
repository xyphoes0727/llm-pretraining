N_VOCAB = 50257  # GPT-2 tokenizer vocab size
MAX_SEQ_LEN = 1024
SEQ_LEN = 1024

N_EPOCHS = 2
BATCH_SIZE = 16
GRAD_ACCUMULATION_STEPS = 8
OPTIM_LR = 1e-4
GRAD_CLIP = 1

MAX_EVAL_BATCHES = 60
DATA_DIRNAME = './data'
DATASET_PATH = './tokenized_data_split'
MODEL_CHECKPOINT_PATH = "./model_checkpoints"
RESUME_FROM_CHECKPOINT = False
RESUME_CHECKPOINT_FILE = "ckpt_0_step_84348.pt"  
MODEL_PARAMS = {
    'd_emb': 768,
    'n_heads': 12,
    'd_val': 64,
    'd_qk': 64,
    'n_blocks': 10
}

NUM_CHECKPOINTS = 32
LOGGING_FREQ = 10
