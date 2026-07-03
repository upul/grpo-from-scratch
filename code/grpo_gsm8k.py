#!/usr/bin/env python
"""
GRPO from scratch on Qwen2.5-1.5B-Instruct + GSM8K.

A hand-written Group Relative Policy Optimization (GRPO) training loop -- no TRL --
that fine-tunes a small instruction model on grade-school math with a verifiable
(0/1 correctness) reward, using LoRA. Every bug-prone mechanic (per-token log-probs,
prompt/padding masking, group-relative advantage, the clipped surrogate + KL) is
implemented explicitly.

Requirements:
    pip install torch transformers peft datasets accelerate matplotlib
    # some environments (e.g. Kaggle) also need:  pip install --upgrade torchao

Usage:
    python grpo_gsm8k.py                     # full run with defaults (tuned for A100 40GB)
    python grpo_gsm8k.py --steps 150         # shorter run
    python grpo_gsm8k.py --group 16          # more samples per prompt (weaker models)
    python grpo_gsm8k.py --diagnostic-only   # sample a few groups, report signal, exit

Outputs (written to --out-dir):
    adapter_step*/   LoRA adapter checkpoints
    history.json     per-step training metrics
    result.json      base/final accuracy + config
    curves.png       reward / signal / stability plots
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Cfg:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    group: int = 8              # completions sampled per prompt (the "group")
    prompts_per_step: int = 4   # prompts per step -> group * prompts sequences/step
    max_new: int = 400          # max completion length
    epochs: int = 2             # update epochs reusing each rollout
    eps: float = 0.2            # PPO/GRPO clip range
    beta: float = 0.04          # KL penalty coefficient
    lr: float = 1e-5
    steps: int = 300
    temp: float = 1.0
    seed: int = 0
    eval_n: int = 100           # problems used for before/after accuracy
    log_every: int = 5
    save_every: int = 50
    out_dir: str = "./grpo_qwen15b_gsm8k"


def parse_args():
    d = Cfg()
    p = argparse.ArgumentParser(
        description="GRPO from scratch on Qwen2.5-1.5B + GSM8K",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=d.model)
    p.add_argument("--group", type=int, default=d.group)
    p.add_argument("--prompts-per-step", type=int, default=d.prompts_per_step)
    p.add_argument("--max-new", type=int, default=d.max_new)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--eps", type=float, default=d.eps)
    p.add_argument("--beta", type=float, default=d.beta)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--steps", type=int, default=d.steps)
    p.add_argument("--temp", type=float, default=d.temp)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--eval-n", type=int, default=d.eval_n)
    p.add_argument("--log-every", type=int, default=d.log_every)
    p.add_argument("--save-every", type=int, default=d.save_every)
    p.add_argument("--out-dir", default=d.out_dir)
    p.add_argument("--diagnostic-only", action="store_true",
                   help="sample a few groups, report mixed/all-correct/all-wrong signal, then exit")
    a = p.parse_args()
    cfg = Cfg(model=a.model, group=a.group, prompts_per_step=a.prompts_per_step,
              max_new=a.max_new, epochs=a.epochs, eps=a.eps, beta=a.beta, lr=a.lr,
              steps=a.steps, temp=a.temp, seed=a.seed, eval_n=a.eval_n,
              log_every=a.log_every, save_every=a.save_every, out_dir=a.out_dir)
    return cfg, a.diagnostic_only


# --------------------------------------------------------------------------- #
# Reward: verifiable 0/1 correctness
# --------------------------------------------------------------------------- #
def extract_answer(text: str):
    """Pull the final numeric answer, reading ####, \\boxed{}, 'final answer', or last number."""
    text = text.replace("{,}", "").replace("\\,", "")           # LaTeX thousands separators
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"\\boxed\{([^}]*)", text)                     # first number inside \boxed{...}
    if m:
        nm = re.search(r"-?\d[\d,]*(?:\.\d+)?", m.group(1))
        if nm:
            return nm.group(0).replace(",", "")
    m = re.search(r"(?:final answer|the answer is)[^\d\-]*(-?[\d,]+(?:\.\d+)?)", text, re.I)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)                 # fallback: last number
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def numbers_equal(a, b) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return str(a).strip() == str(b).strip()


def reward_fn(text: str, gold: str) -> float:
    """Pure verifiable correctness. Clean and unhackable when groups have a right/wrong mix.

    If `--diagnostic-only` shows very few MIXED groups (a weak model), you can swap in a
    small partial-credit shaping term here, e.g.:
        r = 1.0 if numbers_equal(extract_answer(text), gold) else 0.0
        if extract_answer(text) is not None: r += 0.1   # nudge toward a parseable answer
        return r
    """
    return 1.0 if numbers_equal(extract_answer(text), gold) else 0.0


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
SYSTEM = ("You are a helpful math assistant. Solve the problem step by step, then end "
          "your response with exactly one final line:\n#### <number>")


def build_prompt(tok, question: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_data(tok):
    data = load_dataset("openai/gsm8k", "main", split="train")
    prompts = [build_prompt(tok, ex["question"]) for ex in data]
    golds = [ex["answer"].split("####")[-1].strip().replace(",", "") for ex in data]
    return prompts, golds


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_model(cfg: Cfg, device: str, dtype):
    tok = AutoTokenizer.from_pretrained(cfg.model)
    tok.padding_side = "left"                       # left-pad so batched generation aligns
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    policy = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=dtype).to(device)
    policy = get_peft_model(policy, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    policy.print_trainable_parameters()
    policy.gradient_checkpointing_enable()
    policy.enable_input_require_grads()            # required for checkpointing with LoRA
    return tok, policy


# --------------------------------------------------------------------------- #
# GRPO core
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_group(policy, tok, prompt_batch, cfg: Cfg, device: str):
    enc = tok(prompt_batch, return_tensors="pt", padding=True).to(device)
    out = policy.generate(**enc, max_new_tokens=cfg.max_new, do_sample=True,
                          temperature=cfg.temp, top_p=1.0,
                          num_return_sequences=cfg.group,
                          pad_token_id=tok.pad_token_id, use_cache=True)
    return out, enc.attention_mask, enc.input_ids.shape[1]


def per_token_logps(model, input_ids, attention_mask):
    """log pi(token_t) for each next token. Memory-efficient: logit[target] - logsumexp(logits),
    which avoids materializing a full-vocabulary fp32 log_softmax (the usual OOM culprit)."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    sel = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    lse = torch.logsumexp(logits, dim=-1)
    return (sel - lse).float()


def ref_logps(policy, input_ids, attention_mask):
    """Reference = base model with the LoRA adapter disabled (no second copy in memory)."""
    with torch.no_grad(), policy.disable_adapter():
        return per_token_logps(policy, input_ids, attention_mask)


def completion_mask(comp, eos_id):
    """1 for generated tokens up to and including the first EOS, 0 afterwards."""
    is_eos = (comp == eos_id)
    after_eos = (is_eos.cumsum(1) - is_eos.long()) > 0
    return (~after_eos).long()


def build_masks(full_ids, prompt_attn, prompt_len, eos_id, cfg: Cfg):
    cmask = completion_mask(full_ids[:, prompt_len:], eos_id)
    full_attn = torch.cat([prompt_attn.repeat_interleave(cfg.group, 0), cmask], dim=1)
    loss_mask = torch.cat([torch.zeros_like(prompt_attn).repeat_interleave(cfg.group, 0),
                           cmask], dim=1)[:, 1:].float()          # shift to align with logps
    return full_attn, loss_mask


def group_advantages(rewards, n_prompts, cfg: Cfg):
    r = rewards.view(n_prompts, cfg.group)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)
    return adv.view(-1, 1)


def grpo_loss(new_logp, old_logp, rlogp, adv, loss_mask, cfg: Cfg):
    ratio = torch.exp(new_logp - old_logp)
    clipped = torch.clamp(ratio, 1 - cfg.eps, 1 + cfg.eps) * adv
    per_tok = -torch.min(ratio * adv, clipped)
    kl = torch.exp(rlogp - new_logp) - (rlogp - new_logp) - 1     # k3 estimator, >= 0
    per_tok = per_tok + cfg.beta * kl
    return (per_tok * loss_mask).sum() / loss_mask.sum()          # mask out prompt & padding


# --------------------------------------------------------------------------- #
# Eval + diagnostics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_accuracy(policy, tok, prompts, golds, cfg: Cfg, device: str, n=100, seed=1234):
    rng = random.Random(seed)                       # same n problems every call -> fair before/after
    idxs = rng.sample(range(len(prompts)), n)
    correct = 0
    for i in idxs:
        enc = tok(prompts[i], return_tensors="pt").to(device)
        out = policy.generate(**enc, max_new_tokens=cfg.max_new, do_sample=False,
                              pad_token_id=tok.pad_token_id, use_cache=True)
        text = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        correct += int(numbers_equal(extract_answer(text), golds[i]))
    return correct / n


@torch.no_grad()
def diagnose_signal(policy, tok, prompts, golds, cfg: Cfg, device: str, check_steps=8):
    all_correct = mixed = all_wrong = solvable = total = 0
    for _ in range(check_steps):
        idx = random.sample(range(len(prompts)), cfg.prompts_per_step)
        full, _, plen = generate_group(policy, tok, [prompts[i] for i in idx], cfg, device)
        texts = tok.batch_decode(full[:, plen:], skip_special_tokens=True)
        gold_rep = [golds[i] for i in idx for _ in range(cfg.group)]
        rewards = torch.tensor([reward_fn(t, g) for t, g in zip(texts, gold_rep)])
        for row in rewards.view(cfg.prompts_per_step, cfg.group):
            n_ok = int((row >= 1.0).sum())
            total += 1
            if n_ok == cfg.group:
                all_correct += 1
            elif n_ok == 0:
                all_wrong += 1
            else:
                mixed += 1
            if n_ok >= 1:
                solvable += 1
    print(f"groups sampled: {total}")
    print(f"  all-correct (0 signal): {all_correct / total:.0%}")
    print(f"  MIXED  (learning here): {mixed / total:.0%}   <-- GRPO trains on these")
    print(f"  all-wrong  (0 signal):  {all_wrong / total:.0%}")
    print(f"  solvable-groups:        {solvable / total:.0%}")
    print(f"  0adv (dead groups):     {(all_correct + all_wrong) / total:.0%}")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def log_metrics(history, step, rewards, loss_mask, old_logp, new_logp, rlogp, cfg: Cfg, t0):
    with torch.no_grad():
        ratio = torch.exp(new_logp - old_logp)
        m = loss_mask
        clipfrac = (((ratio < 1 - cfg.eps) | (ratio > 1 + cfg.eps)).float() * m).sum() / m.sum()
        kl = ((torch.exp(rlogp - new_logp) - (rlogp - new_logp) - 1) * m).sum() / m.sum()
        comp_len = m.sum(1).float().mean()
        rg = rewards.view(cfg.prompts_per_step, cfg.group)
        zero_adv = (rg.std(1) < 1e-6).float().mean()
        solve = (rewards >= 1.0).float().mean()
        solvable = (rg >= 1.0).any(1).float().mean()
    rec = dict(step=step, reward=rewards.mean().item(), kl=kl.item(), clipfrac=clipfrac.item(),
               comp_len=comp_len.item(), zero_adv=zero_adv.item(),
               solve=solve.item(), solvable=solvable.item())
    history.append(rec)
    sps = (time.time() - t0) / (step + 1)
    print(f"step {step:4d} | reward {rec['reward']:.3f} | solve {rec['solve']:.2f} "
          f"| solvable {rec['solvable']:.2f} | kl {rec['kl']:.4f} | clipfrac {rec['clipfrac']:.3f} "
          f"| len {rec['comp_len']:.0f} | 0adv {rec['zero_adv']:.2f} "
          f"| {sps:.1f}s/step | eta {sps * cfg.steps / 60:.0f}m")


def save_ckpt(policy, history, cfg: Cfg, step):
    path = os.path.join(cfg.out_dir, f"adapter_step{step}")
    policy.save_pretrained(path)
    with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"  saved adapter + history at step {step}")


def train(policy, tok, prompts, golds, cfg: Cfg, device: str):
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=cfg.lr)
    eos_id = tok.eos_token_id
    history = []
    t0 = time.time()

    for step in range(cfg.steps):
        # 1. ROLLOUT
        idx = random.sample(range(len(prompts)), cfg.prompts_per_step)
        full, prompt_attn, prompt_len = generate_group(
            policy, tok, [prompts[i] for i in idx], cfg, device)
        full_attn, loss_mask = build_masks(full, prompt_attn, prompt_len, eos_id, cfg)

        # 2. SCORE -> group-relative advantage
        texts = tok.batch_decode(full[:, prompt_len:], skip_special_tokens=True)
        gold_rep = [golds[i] for i in idx for _ in range(cfg.group)]
        rewards = torch.tensor([reward_fn(t, g) for t, g in zip(texts, gold_rep)], device=device)
        adv = group_advantages(rewards, cfg.prompts_per_step, cfg)

        # 3. SNAPSHOT generating-policy + reference log-probs
        with torch.no_grad():
            old_logp = per_token_logps(policy, full, full_attn)
        rlogp = ref_logps(policy, full, full_attn)

        # 4-5. UPDATE (reuse the batch for a few epochs)
        for epoch in range(cfg.epochs):
            new_logp = per_token_logps(policy, full, full_attn)
            if epoch == 0:
                r0 = torch.exp(new_logp - old_logp)
                assert ((r0 - 1).abs() * loss_mask).max() < 1e-3, \
                    "epoch-0 ratio != 1 -> masking bug!"
            loss = grpo_loss(new_logp, old_logp, rlogp, adv, loss_mask, cfg)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        if step % cfg.log_every == 0:
            with torch.no_grad():
                new_logp = per_token_logps(policy, full, full_attn)
            log_metrics(history, step, rewards, loss_mask, old_logp, new_logp, rlogp, cfg, t0)
            print(f"  sample: gold={gold_rep[0]} extracted={extract_answer(texts[0])}")
        if step and step % cfg.save_every == 0:
            save_ckpt(policy, history, cfg, step)

    save_ckpt(policy, history, cfg, cfg.steps)
    return history


def plot_curves(history, base_acc, final_acc, cfg: Cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = [h["step"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(steps, [h["reward"] for h in history])
    ax[0].set_title("mean reward = solve rate"); ax[0].set_xlabel("step")
    ax[1].plot(steps, [h["solvable"] for h in history], label="solvable groups")
    ax[1].legend(); ax[1].set_title("correctness signal"); ax[1].set_xlabel("step")
    ax[2].plot(steps, [h["kl"] for h in history], label="KL to ref")
    ax[2].plot(steps, [h["clipfrac"] for h in history], label="clip fraction")
    ax[2].legend(); ax[2].set_title("stability"); ax[2].set_xlabel("step")
    title = "GRPO on Qwen2.5-1.5B + GSM8K"
    if base_acc is not None and final_acc is not None:
        title += f"  |  accuracy {base_acc:.0%} -> {final_acc:.0%}"
    plt.suptitle(title)
    plt.tight_layout()
    out = os.path.join(cfg.out_dir, "curves.png")
    plt.savefig(out, dpi=120)
    print(f"saved plot -> {out}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    cfg, diagnostic_only = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    assert torch.cuda.is_available(), "This script requires a CUDA GPU."
    device = "cuda:0"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)} | dtype: {dtype}")

    tok, policy = load_model(cfg, device, dtype)
    prompts, golds = load_data(tok)
    print(f"{len(prompts)} training prompts")

    if diagnostic_only:
        diagnose_signal(policy, tok, prompts, golds, cfg, device)
        return

    base_acc = eval_accuracy(policy, tok, prompts, golds, cfg, device, n=cfg.eval_n)
    print(f"BASE accuracy ({cfg.eval_n} problems, greedy): {base_acc:.1%}")

    history = train(policy, tok, prompts, golds, cfg, device)

    final_acc = eval_accuracy(policy, tok, prompts, golds, cfg, device, n=cfg.eval_n)
    print("=" * 54)
    print(f"GSM8K accuracy  BEFORE {base_acc:.1%}  ->  AFTER {final_acc:.1%}  "
          f"({(final_acc - base_acc) * 100:+.1f} pts)")
    print("=" * 54)

    with open(os.path.join(cfg.out_dir, "result.json"), "w") as f:
        json.dump({"base_acc": base_acc, "final_acc": final_acc,
                   "improvement_points": (final_acc - base_acc) * 100,
                   "config": asdict(cfg)}, f, indent=2)
    plot_curves(history, base_acc, final_acc, cfg)


if __name__ == "__main__":
    main()
