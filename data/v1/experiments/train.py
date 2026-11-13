"""Fine-tune Qwen3-0.6B-Base on dataset v1 with LoRA: the CLiC-it 2026 recipe with v1's control strings.

Recipe of GFFF/controlled_text_simplification_ft.py: LoRA r 32, alpha 64, dropout 0.05 on
q/k/v/o/gate/up/down, embed_tokens and lm_head trained in full (modules_to_save), learning rate 2.5e-5
with linear decay and 2 warm-up steps, effective batch 64 (GFFF: 16 x 4; here 4 x 16 by default), gradient
checkpointing, 3 epochs, bf16, fused AdamW,
prompt f"{ctrl}\\n{original}\\n", loss on the simplification only, control tokens added as special
tokens, embeddings resized with pad_to_multiple_of=8.

Changes from that script:
- the new tokens are mean-initialised by token id (the GFFF code missed the first ones);
- validation is v1's `split` column (the GFFF code took the last rows by idx, sharing originals);
- each set target is replaced by "any" with probability 0.3, re-drawn every time an example is
  read (so every epoch); at least one target is kept; source tokens are never dropped
  (--target-dropout 0 turns this off, for data already in one-token-per-example form, dataset v7);
- an example longer than 256 tokens loses the end of its target, not its start (the GFFF code cut
  from the left, which could remove the control tokens);
- dynamic padding (speed only; the loss is the same).

Options for later runs (their defaults reproduce every earlier run): --epochs (default 3), --eval-steps (default 500)
and --keep-best, which also saves the adapter at the lowest validation loss to <out>/adapter_best (the step is in
<out>/adapter_best/best.json); <out>/adapter is always the last step.

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python data/v1/experiments/train.py --variant targets

Dataset v2 (same recipe, same evaluation; see data/v2/README.md):

    ... train.py --variant targets --data data/v2/training_set_v2.zip --out data/v1/experiments/runs/v2_targets
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch allocates

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

from common import BASE_MODEL, DIMS, RUNS, SEED, V1, VARIANTS, control_string, control_tokens, prompt

TARGET_DROPOUT = 0.3
MAX_LENGTH = 256
# Effective batch 64 as in GFFF (16 x 4); the per-device split is set by --per-device-batch so that
# training fits next to other users of GPU0 (default 4 x 16, with gradient checkpointing as in GFFF).
EFFECTIVE_BATCH = 64
HYPER = dict(learning_rate=2.5e-5, per_device_train_batch_size=4, gradient_accumulation_steps=16, num_train_epochs=3,
             warmup_steps=2, lora_r=32, lora_alpha=64, lora_dropout=0.05)


class V1Pairs(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, variant: str, dropout: float, seed: int):
        cols = ["original_text", "simplification"] + [f"tgt_{d}" for d in DIMS] + [f"src_{d}" for d in DIMS]
        self.rows = frame[cols].to_dict("records")
        self.tok, self.variant, self.dropout, self.rng = tokenizer, variant, dropout, random.Random(seed)
        self.truncated = 0

    def __len__(self) -> int:
        return len(self.rows)

    def control(self, r: dict) -> str:
        targets = {d: r[f"tgt_{d}"] for d in DIMS}
        if self.dropout > 0:
            set_dims = [d for d in DIMS if targets[d] != "any"]
            keep = [d for d in set_dims if self.rng.random() >= self.dropout] or [self.rng.choice(set_dims)]
            targets = {d: targets[d] if d in keep else "any" for d in DIMS}
        return control_string(self.variant, targets, {d: r[f"src_{d}"] for d in DIMS})

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        p_ids = self.tok(prompt(self.control(r), r["original_text"]), add_special_tokens=False)["input_ids"]
        a_ids = self.tok(r["simplification"].strip() + self.tok.eos_token, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(a_ids) > MAX_LENGTH:
            self.truncated += 1
            a_ids = a_ids[: max(1, MAX_LENGTH - len(p_ids))]
        return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids}


def collate(pad_id: int):
    def fn(batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k], labels[i, :k], mask[i, :k] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}
    return fn


class KeepBest(TrainerCallback):
    """Save the adapter whenever the validation loss reaches a new minimum."""

    def __init__(self, folder: Path, tokenizer):
        self.folder, self.tokenizer, self.best = folder, tokenizer, float("inf")

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        loss = (metrics or {}).get("eval_loss")
        if loss is None or model is None or loss >= self.best:
            return
        self.best = loss
        model.save_pretrained(self.folder)
        self.tokenizer.save_pretrained(self.folder)
        (self.folder / "best.json").write_text(json.dumps({"step": state.global_step, "eval_loss": loss}))
        print(f"[keep-best] step {state.global_step}: validation loss {loss:.4f}, saved {self.folder}", flush=True)


class Timing(TrainerCallback):
    def __init__(self):
        self.t0 = None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            self.t0 = time.time()
        if state.global_step == 51 and self.t0:
            per_step = (time.time() - self.t0) / 50
            print(f"[timing] {per_step:.2f}s/step -> about {per_step * state.max_steps / 3600:.1f}h for {state.max_steps} steps", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--out", help="run folder (default: runs/<variant>)")
    parser.add_argument("--data", type=Path, default=V1 / "training_set_v1.zip", help="training set (default: dataset v1)")
    parser.add_argument("--max-steps", type=int, default=-1, help="for smoke tests")
    parser.add_argument("--limit", type=int, help="use only the first N training pairs (smoke tests)")
    parser.add_argument("--per-device-batch", type=int, default=4, help="examples per step; accumulation keeps the batch at 64")
    parser.add_argument("--target-dropout", type=float, default=TARGET_DROPOUT, help="probability of dropping each set target (0: off)")
    parser.add_argument("--epochs", type=float, default=HYPER["num_train_epochs"], help="training epochs (default 3)")
    parser.add_argument("--eval-steps", type=int, default=500, help="steps between validation losses (default 500)")
    parser.add_argument("--keep-best", action="store_true", help="also save the adapter at the lowest validation loss to <out>/adapter_best")
    args = parser.parse_args()
    HYPER["num_train_epochs"] = args.epochs
    HYPER["per_device_train_batch_size"] = args.per_device_batch
    HYPER["gradient_accumulation_steps"] = EFFECTIVE_BATCH // args.per_device_batch
    out = RUNS / args.variant if args.out is None else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)

    data = pd.read_csv(args.data, usecols=["idx", "split", "original_text", "simplification"]
                       + [f"tgt_{d}" for d in DIMS] + [f"src_{d}" for d in DIMS], dtype={f"tgt_{d}": str for d in DIMS})
    train, valid = data[data.split == "train"], data[data.split == "validation"]
    if args.limit:
        train, valid = train.head(args.limit), valid.head(max(8, args.limit // 8))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_before = len(tokenizer)
    tokens = control_tokens(args.variant)
    added = tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    assert added == len(tokens), f"added {added} of {len(tokens)} tokens"

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        emb[ids] = emb[:n_before].mean(dim=0, keepdim=True)
        out_emb = model.get_output_embeddings()
        if out_emb is not None and out_emb.weight.data_ptr() != emb.data_ptr():
            out_emb.weight[ids] = out_emb.weight[:n_before].mean(dim=0, keepdim=True)

    model = get_peft_model(model, LoraConfig(
        r=HYPER["lora_r"], lora_alpha=HYPER["lora_alpha"], lora_dropout=HYPER["lora_dropout"], bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"]))
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.enable_input_require_grads()

    train_ds = V1Pairs(train, tokenizer, args.variant, args.target_dropout, SEED)
    valid_ds = V1Pairs(valid, tokenizer, args.variant, 0.0, SEED)
    print("example:", tokenizer.decode(train_ds[0]["input_ids"])[:400].replace("\n", " | "), flush=True)

    targs = TrainingArguments(
        output_dir=str(out), seed=SEED, data_seed=SEED, bf16=True, optim="adamw_torch_fused",
        learning_rate=HYPER["learning_rate"], lr_scheduler_type="linear", warmup_steps=HYPER["warmup_steps"],
        per_device_train_batch_size=HYPER["per_device_train_batch_size"],
        gradient_accumulation_steps=HYPER["gradient_accumulation_steps"], num_train_epochs=HYPER["num_train_epochs"],
        max_steps=args.max_steps, per_device_eval_batch_size=16, eval_strategy="steps", eval_steps=args.eval_steps,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=50, save_strategy="no", report_to=[], remove_unused_columns=False, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=valid_ds,
                      data_collator=collate(tokenizer.pad_token_id),
                      callbacks=[Timing()] + ([KeepBest(out / "adapter_best", tokenizer)] if args.keep_best else []))
    t0 = time.time()
    trainer.train()
    final_eval = trainer.evaluate()

    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    (out / "training.json").write_text(json.dumps({
        "variant": args.variant, "data": str(args.data), "base_model": BASE_MODEL, "hyperparameters": HYPER, "target_dropout": args.target_dropout,
        "max_length": MAX_LENGTH, "train_pairs": len(train_ds), "validation_pairs": len(valid_ds),
        "new_tokens": len(tokens), "steps": trainer.state.global_step, "hours": round((time.time() - t0) / 3600, 2),
        "examples_truncated_reads": train_ds.truncated, "final_eval": final_eval, "eval_steps": args.eval_steps,
        "best": json.loads((out / "adapter_best" / "best.json").read_text()) if args.keep_best else None,
        "log_history": trainer.state.log_history,
    }, indent=1))
    print(f"saved {out / 'adapter'}", flush=True)


if __name__ == "__main__":
    main()
