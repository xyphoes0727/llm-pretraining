import time
import torch
import logging
import config
import math
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


def log_train_metrics(epoch: int, n_steps: int, loss: float, t1: float, last_lr: float):
    logger.info(
        f"Epoch: {epoch} Model training loss at step {n_steps} is {loss:.4f}. Perp: {math.exp(loss):.4f} LR: {last_lr}")
    logger.info(f"{n_steps*config.BATCH_SIZE*config.GRAD_ACCUMULATION_STEPS} datapoints processed")
    if (t1 == 0.):
        return

    logger.info(
        f"Time passed for {config.LOGGING_FREQ} steps: {time.time()- t1}")


def log_eval_metrics(epoch: int, n_steps: int, loss: float):
    logger.info(
        f"Model training loss at step {n_steps} is {loss} Perp: {math.exp(loss):.4f}\n")
    logger.info(
        f" Epoch {epoch}: {n_steps*config.BATCH_SIZE*config.GRAD_ACCUMULATION_STEPS} samples processed. Evaluating model...\n")


def save_model(model_state_dict: dict,
               opt_state_dict: dict, scaler_state_dict: dict,
               epoch: int, step: int):
    torch.save({
        "model": model_state_dict,
        "optimizer": opt_state_dict,
        "scaler_state_dict": scaler_state_dict,
        "step": epoch
    }, f"{config.MODEL_CHECKPOINT_PATH}/ckpt_{epoch}_step_{step}.pt")
