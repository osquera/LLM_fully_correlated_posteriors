"""Stage 1: obtain theta_map by LoRA fine-tuning SmolLM2 on a small corpus.

The projection is post-hoc: it needs a point estimate *and* the training set to
project against. SmolLM2's pretraining corpus is out of reach, so we fine-tune
on a small task dataset and build the posterior with respect to *that* set.
This is also what makes the paper's central claim testable -- the projected
posterior must preserve the fine-tuning loss.

Only the LoRA adapters are trainable, so theta has P ~ 1-5M rather than 135M.
That keeps every sample vector small enough to hold several at once, and puts
the work in direct comparison with Laplace-LoRA (Yang et al., 2024).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--dataset", default="tatsu-lab/alpaca")
    ap.add_argument("--split", default="train[:2000]")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument(
        "--lora-targets", nargs="+", default=["q_proj", "v_proj"],
        help="Modules to adapt. Widen to k_proj/o_proj/gate_proj/... to grow P.",
    )
    ap.add_argument("--out", default="checkpoints/smollm2_lora")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def build_loader(args, tokenizer):
    ds = load_dataset(args.dataset, split=args.split)
    field = args.text_field if args.text_field in ds.column_names else ds.column_names[0]

    def tok(batch):
        return tokenizer(
            batch[field],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    ds = ds.map(tok, batched=True, remove_columns=ds.column_names)
    ds.set_format("torch", columns=["input_ids", "attention_mask"])
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True), ds


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # eager attention: torch.func.jvp (forward-mode AD) does not compose with
    # the fused sdpa / flash kernels, and the projection needs JVPs later.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager", dtype=torch.float32
    ).to(args.device)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_targets,
        lora_dropout=0.0,  # must be 0: the projection assumes a deterministic f
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    loader, ds = build_loader(args, tokenizer)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total, n = 0.0, 0
        for batch in tqdm(loader, desc=f"epoch {epoch}"):
            ids = batch["input_ids"].to(args.device)
            mask = batch["attention_mask"].to(args.device)
            out = model(input_ids=ids, attention_mask=mask, labels=ids)
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            total += out.loss.item() * ids.size(0)
            n += ids.size(0)
        print(f"epoch {epoch}: train loss = {total / n:.4f}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)
    (outdir / "run.json").write_text(json.dumps(vars(args), indent=2))
    print(f"saved theta_map to {outdir}")


if __name__ == "__main__":
    main()
