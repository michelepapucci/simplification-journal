"""Fine-tune Qwen3-0.6B-Base on dataset g1 with LoRA: one control token on READ-IT global.

The recipe of the journal's earlier models (archived on the multidim-archive branch), itself the CLiC-it 2026 one:
LoRA r 32, alpha 64, dropout 0.05 on q/k/v/o/gate/up/down, embed_tokens and lm_head trained in full
(modules_to_save), learning rate 2.5e-5 with linear decay and 2 warm-up steps, effective batch 64 (4 x 16 here, with
gradient checkpointing), 3 epochs, bf16, fused AdamW, prompt f"{token}\\n{original}\\n", loss on the
simplification only, new tokens mean-initialised, an example over 256 tokens loses the end of its target.
Validation is the dataset's `split` column (1% of the originals). The final adapter is the model.

Two options, one per model variant:
- --balance: every level 0-70 is shown as often as the most frequent one (level 80, which no test original can
  request, is left as it is). Each epoch shows every pair once, and tops up the rarer levels with pairs of that
  level drawn at random every time an example is read, so the extra copies change from epoch to epoch.
- --null-share p: each example gets <|readability_any|> instead of its level with probability p, re-drawn every
  time it is read; validation examples always keep their level. This teaches the model a "no level" prompt for
  classifier-free guidance (generate.py --guidance).

    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 9-25 python experiments/train.py --name g1
    ... --name g1_balanced --balance
    ... --name g1_guided --null-share 0.1
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

from common import BASE_MODEL, G1, NULL, RUNS, SEED, TOKENS, prompt, token

MAX_LENGTH = 256
EFFECTIVE_BATCH = 64
BALANCED_LEVELS = tuple(range(0, 80, 10))
HYPER = dict(learning_rate=2.5e-5, per_device_train_batch_size=4, gradient_accumulation_steps=16, num_train_epochs=3,
             warmup_steps=2, lora_r=32, lora_alpha=64, lora_dropout=0.05)


class G1Pairs(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, balance: bool, null_share: float, seed: int):
        self.rows = frame[["original_text", "simplification", "tgt_level"]].to_dict("records")
        self.tok, self.null_share, self.rng = tokenizer, null_share, random.Random(seed)
        self.truncated, self.nulls = 0, 0
        by_level = frame.reset_index(drop=True).groupby("tgt_level").groups
        self.by_level = {int(lv): list(ix) for lv, ix in by_level.items()}
        if balance:
            cap = max(len(self.by_level.get(lv, [])) for lv in BALANCED_LEVELS)
            self.index = [(lv, k) for lv, ix in self.by_level.items()
                          for k in range(max(len(ix), cap) if lv in BALANCED_LEVELS else len(ix))]
        else:
            self.index = [(None, k) for k in range(len(self.rows))]

    def __len__(self) -> int:
        return len(self.index)

    def shown_per_level(self) -> dict:
        out = {}
        for lv, k in self.index:
            lv = self.rows[k]["tgt_level"] if lv is None else lv
            out[int(lv)] = out.get(int(lv), 0) + 1
        return dict(sorted(out.items()))

    def __getitem__(self, i: int) -> dict:
        lv, k = self.index[i]
        if lv is None:
            r = self.rows[k]
        else:
            ix = self.by_level[lv]
            r = self.rows[ix[k]] if k < len(ix) else self.rows[self.rng.choice(ix)]
        if self.null_share > 0 and self.rng.random() < self.null_share:
            ctrl = token(NULL)
            self.nulls += 1
        else:
            ctrl = token(r["tgt_level"])
        p_ids = self.tok(prompt(ctrl, r["original_text"]), add_special_tokens=False)["input_ids"]
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
    parser.add_argument("--name", required=True, help="run folder under runs/")
    parser.add_argument("--data", type=Path, default=G1 / "training_set_g1.zip")
    parser.add_argument("--balance", action="store_true", help="show every level 0-70 as often as the most frequent one")
    parser.add_argument("--null-share", type=float, default=0.0, help="probability of the 'no level' token (guided model: 0.1)")
    parser.add_argument("--epochs", type=float, default=HYPER["num_train_epochs"])
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=-1, help="for smoke tests")
    parser.add_argument("--limit", type=int, help="use only the first N training pairs (smoke tests)")
    parser.add_argument("--per-device-batch", type=int, default=4, help="examples per step; accumulation keeps the batch at 64")
    args = parser.parse_args()
    HYPER["num_train_epochs"] = args.epochs
    HYPER["per_device_train_batch_size"] = args.per_device_batch
    HYPER["gradient_accumulation_steps"] = EFFECTIVE_BATCH // args.per_device_batch
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)

    data = pd.read_csv(args.data, usecols=["idx", "split", "original_text", "simplification", "tgt_level"])
    train, valid = data[data.split == "train"], data[data.split == "validation"]
    if args.limit:
        train, valid = train.head(args.limit), valid.head(max(8, args.limit // 8))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_before = len(tokenizer)
    added = tokenizer.add_special_tokens({"additional_special_tokens": TOKENS})
    assert added == len(TOKENS), f"added {added} of {len(TOKENS)} tokens"

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    ids = tokenizer.convert_tokens_to_ids(TOKENS)
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

    train_ds = G1Pairs(train, tokenizer, args.balance, args.null_share, SEED)
    valid_ds = G1Pairs(valid, tokenizer, False, 0.0, SEED)
    shown = train_ds.shown_per_level()
    print(f"examples per epoch: {len(train_ds)} (pairs {len(train)}); per level: {shown}", flush=True)
    print("example:", tokenizer.decode(train_ds[0]["input_ids"])[:300].replace("\n", " | "), flush=True)

    targs = TrainingArguments(
        output_dir=str(out), seed=SEED, data_seed=SEED, bf16=True, optim="adamw_torch_fused",
        learning_rate=HYPER["learning_rate"], lr_scheduler_type="linear", warmup_steps=HYPER["warmup_steps"],
        per_device_train_batch_size=HYPER["per_device_train_batch_size"],
        gradient_accumulation_steps=HYPER["gradient_accumulation_steps"], num_train_epochs=HYPER["num_train_epochs"],
        max_steps=args.max_steps, per_device_eval_batch_size=16, eval_strategy="steps", eval_steps=args.eval_steps,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=50, save_strategy="no", report_to=[], remove_unused_columns=False, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=valid_ds,
                      data_collator=collate(tokenizer.pad_token_id), callbacks=[Timing()])
    t0 = time.time()
    trainer.train()
    final_eval = trainer.evaluate()

    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    (out / "training.json").write_text(json.dumps({
        "name": args.name, "data": str(args.data), "base_model": BASE_MODEL, "hyperparameters": HYPER,
        "balance": args.balance, "null_share": args.null_share, "max_length": MAX_LENGTH,
        "train_pairs": len(train), "examples_per_epoch": len(train_ds), "examples_per_level_per_epoch": shown,
        "validation_pairs": len(valid_ds), "new_tokens": len(TOKENS), "steps": trainer.state.global_step,
        "hours": round((time.time() - t0) / 3600, 2), "examples_truncated_reads": train_ds.truncated,
        "null_token_reads": train_ds.nulls, "final_eval": final_eval, "eval_steps": args.eval_steps,
        "log_history": trainer.state.log_history,
    }, indent=1))
    print(f"saved {out / 'adapter'}", flush=True)


if __name__ == "__main__":
    main()
