import argparse
import os
from typing import List, Optional

import torch
from tokenizers import Tokenizer

import config
from model import MyModel


def resolve_checkpoint_path(path: Optional[str]) -> str:
    if path:
        if os.path.isabs(path):
            return path
        return os.path.join(config.MODEL_CHECKPOINT_PATH, path)

    # Default behavior mirrors training resume checkpoint.
    if config.RESUME_CHECKPOINT_FILE:
        return os.path.join(config.MODEL_CHECKPOINT_PATH, config.RESUME_CHECKPOINT_FILE)

    raise ValueError(
        "No checkpoint provided. Pass --checkpoint or set RESUME_CHECKPOINT_FILE in config.py"
    )


def load_model(checkpoint_path: str, device: torch.device) -> MyModel:
    model = MyModel(**config.MODEL_PARAMS).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def top_k_sample(logits: torch.Tensor, top_k: int, temperature: float) -> int:
    if temperature <= 0:
        # Greedy decode fallback
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        topk_values, topk_indices = torch.topk(logits, k)
        probs = torch.softmax(topk_values, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1)
        token_id = topk_indices.gather(-1, sampled_idx)
        return int(token_id.item())

    probs = torch.softmax(logits, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1)
    return int(token_id.item())


def generate_ids(
    model: MyModel,
    input_ids: List[int],
    max_new_tokens: int,
    eos_id: Optional[int],
    top_k: int,
    temperature: float,
    device: torch.device,
) -> List[int]:
    ids = list(input_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if len(ids) >= config.MAX_SEQ_LEN:
                break

            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(x)
            next_token_logits = logits[0, -1, :]
            next_id = top_k_sample(next_token_logits, top_k=top_k, temperature=temperature)
            ids.append(next_id)

            if eos_id is not None and next_id == eos_id:
                break

    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint filename or absolute path")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (0 disables top-k)")
    args = parser.parse_args()

    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"Tokenizer file not found: {args.tokenizer}")

    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    encoded = tokenizer.encode(args.prompt)
    input_ids = encoded.ids

    # Training data includes <bos> ... <eos>; for continuation we strip terminal <eos>.
    if eos_id is not None and input_ids and input_ids[-1] == eos_id:
        input_ids = input_ids[:-1]

    if not input_ids:
        if bos_id is None:
            raise ValueError("Prompt produced empty token list and tokenizer has no <bos> token.")
        input_ids = [bos_id]

    if len(input_ids) >= config.MAX_SEQ_LEN:
        input_ids = input_ids[-(config.MAX_SEQ_LEN - 1):]

    model = load_model(ckpt_path, device)

    generated_ids = generate_ids(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_id=eos_id,
        top_k=args.top_k,
        temperature=args.temperature,
        device=device,
    )

    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    prompt_text = tokenizer.decode(input_ids, skip_special_tokens=True)

    if full_text.startswith(prompt_text):
        completion_text = full_text[len(prompt_text):]
    else:
        completion_text = full_text

    print("=== Prompt ===")
    print(args.prompt)
    print("\n=== Completion ===")
    print(completion_text)
    print("\n=== Full Text ===")
    print(full_text)


if __name__ == "__main__":
    main()
