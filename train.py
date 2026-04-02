import time
from helpers import log_train_metrics, log_eval_metrics, save_model
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
import re

load_dotenv(".env")

torch.manual_seed(42)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

wandb.login(key=os.getenv("WANDB_KEY"))

run = wandb.init(
    entity="xyphoes-iit-jodhpur",
    project="llm_training_2",
    config=config.MODEL_PARAMS,
)

model = MyModel(**config.MODEL_PARAMS).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.OPTIM_LR)

# Required for mixed precision training to prevent underflow due to lower precision
scaler = torch.amp.GradScaler()

n_steps = 0
if config.RESUME_FROM_CHECKPOINT:
    if not config.RESUME_CHECKPOINT_FILE:
        raise ValueError(
            "RESUME_FROM_CHECKPOINT is True, but RESUME_CHECKPOINT_FILE is empty."
        )

    ckpt_path = os.path.join(
        config.MODEL_CHECKPOINT_PATH, config.RESUME_CHECKPOINT_FILE)
    checkpoint = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    elif "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    step_match = re.search(r"_step_(\d+)\.pt$", config.RESUME_CHECKPOINT_FILE)
    if step_match:
        n_steps = int(step_match.group(1))

    logger.info(f"Resumed from checkpoint: {ckpt_path}")
    logger.info(f"Continuing training from global step: {n_steps}")

train_data_loader, test_data_loader = load_dataset(config.DATASET_PATH)

# T_Max is the time period in which LR will decrease each step, starting from 1e-3
# We want it to anneal throughout both epochs, and also for each step, but our step is defined as grad_accumulation step
steps_per_epoch = len(train_data_loader) // config.GRAD_ACCUMULATION_STEPS
T_max = config.N_EPOCHS * steps_per_epoch

# Continue LR from where it was prev. left off
for param_group in optimizer.param_groups:
    param_group.setdefault("initial_lr", param_group["lr"])

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                          T_max,
                                                          last_epoch=max(-1, n_steps - 1),
                                                          eta_min=1e-5)

# Ignore loss of padding tokens using ignore_index
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

logger.info(f"Starting training...")
model.train()
steps_per_checkpoint = steps_per_epoch * \
    config.N_EPOCHS // config.NUM_CHECKPOINTS

for epoch in range(config.N_EPOCHS):
    logger.info(f"Starting epoch {epoch}.")
    t1 = 0.
    step_loss_sum = 0.0
    for i, data in enumerate(train_data_loader):
        data = data.to(device=device, non_blocking=True)
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # Here, we are taking inputs and our model outputs in such a way that we get the next most likely
            # token. So, each i'th index in (B,n,N_VOCAB) o/p will be the probs of the i+1'th

            # input will be 1s to n-1th token. Target will be 2nd to nth token
            inputs = data[:, :-1]
            targets = data[:, 1:]
            out = model(inputs)

            logits = out.view(-1, config.N_VOCAB)
            targets = targets.contiguous().view(-1)

            raw_loss = loss_fn(logits, targets)
            step_loss_sum += raw_loss.item()
            loss = raw_loss / config.GRAD_ACCUMULATION_STEPS
            # Scales loss, then computes backward pass
            scaler.scale(loss).backward()

        if ((i+1) % config.GRAD_ACCUMULATION_STEPS == 0):
            n_steps += 1
            train_step_loss = step_loss_sum / config.GRAD_ACCUMULATION_STEPS
            step_loss_sum = 0.0
            # To clip gradients, we HAVE to unscale the gradients first
            scaler.unscale_(optimizer)
            clip_grad.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

            # optimizer will be stepped using scaler now.
            scaler.step(optimizer)
            lr_scheduler.step()
            optimizer.zero_grad()
            scaler.update()

            if (n_steps % config.LOGGING_FREQ == 0):
                log_train_metrics(epoch, n_steps, train_step_loss, t1, lr_scheduler.get_last_lr()[0])
                run.log({
                    "epoch": epoch,
                    "train_loss": train_step_loss,
                    "train_perp": math.exp(train_step_loss),
                    "step": n_steps,
                    "lr": lr_scheduler.get_last_lr()[0]
                })
                t1 = time.time()

            if (n_steps % steps_per_checkpoint == 0 and n_steps > 0):
                log_eval_metrics(epoch, n_steps, train_step_loss)
                model.eval()
                total_loss = 0.
                with (torch.no_grad()):
                    for i, data in enumerate(test_data_loader):
                        if (i >= config.MAX_EVAL_BATCHES):
                            break

                        data = data.to(device=device, non_blocking=True)
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            inputs = data[:, :-1]
                            targets = data[:, 1:]
                            out = model(inputs)

                            logits = out.view(-1, config.N_VOCAB)
                            targets = targets.contiguous().view(-1)

                            loss = loss_fn(logits, targets)
                            total_loss += loss.item()

                test_loss = total_loss/config.MAX_EVAL_BATCHES
                logger.info(
                    f"Model test loss at step {n_steps} is {test_loss:.4f}, Perp: {math.exp(test_loss)} \n")

                run.log({
                    "epoch": epoch,
                    "test_loss": test_loss,
                    "test_perp": math.exp(test_loss),
                    "step": n_steps,
                })
                save_model(model.state_dict(),
                        optimizer.state_dict(), scaler.state_dict(), epoch, n_steps)
                model.train()

logger.info(f"Finished training! Saving final model...")
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scaler": scaler.state_dict(),
}, f"{config.MODEL_CHECKPOINT_PATH}/Final_Model.pt")

run.finish()
