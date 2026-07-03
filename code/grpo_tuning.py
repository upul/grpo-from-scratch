#!/usr/bin/env python
"""
Instrumented GRPO on a toy task -- the evidence behind guides/grpo-tuning-guide.md.

Runs a healthy baseline with the full metrics dashboard (reward, KL, clip fraction,
entropy, grad norm, zero-advantage rate), then sweeps each hyperparameter so you can
see -- from real numbers -- what each knob does.

The absolute numbers are toy-specific; the *shapes* (too-high BETA suppresses learning,
too-high LR adds instability, small groups give a noisy baseline, more epochs raise the
clip fraction) are what transfer to any GRPO run.

Run:  python grpo_tuning.py
"""
import copy
import random
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- task: reward = count of {a, e} in a 12-letter vocab (harder than the 10-letter toy,
#             so hyperparameters actually matter) -----
CHARS = list("abcdefghijkl")
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB = len(CHARS)
PROMPT = [STOI["a"]]
GEN_LEN = 8
TARGET = set("ae")


def reward_fn(toks):
    return float(sum(1 for t in toks if ITOS[t] in TARGET))


class TinyGPT(nn.Module):
    def __init__(self, vocab=VOCAB, d=32, heads=2, ctx=16):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.head = nn.Linear(d, vocab)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T))
        h = self.ln1(x)
        m = torch.triu(torch.ones(T, T) * float("-inf"), diagonal=1)
        a, _ = self.attn(h, h, h, attn_mask=m)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return self.head(x)


def generate(model, n):
    seq = torch.tensor(PROMPT).repeat(n, 1)
    with torch.no_grad():
        for _ in range(GEN_LEN):
            p = F.softmax(model(seq)[:, -1, :], dim=-1)
            seq = torch.cat([seq, torch.multinomial(p, 1)], dim=1)
    return seq


def logps_ent(model, seq):
    """Per-token log-probs (n, GEN_LEN) and per-token entropy (n, GEN_LEN)."""
    lp = F.log_softmax(model(seq[:, :-1]), dim=-1)
    tok = lp.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)[:, -GEN_LEN:]
    ent = -(lp.exp() * lp).sum(-1)[:, -GEN_LEN:]
    return tok, ent


def run_grpo(lr=3e-3, GROUP=8, EPOCHS=4, EPS=0.2, BETA=0.02, STEPS=50, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    policy = TinyGPT()
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    hist = []
    for step in range(STEPS):
        seq = generate(policy, GROUP)
        comps = seq[:, 1:].tolist()
        rewards = torch.tensor([reward_fn(c) for c in comps])
        rstd = rewards.std()
        adv = ((rewards - rewards.mean()) / (rstd + 1e-4)).unsqueeze(1)
        with torch.no_grad():
            old, _ = logps_ent(policy, seq)
            rl, _ = logps_ent(ref, seq)

        clipfracs, gnorms = [], []
        for _ in range(EPOCHS):
            new, ent = logps_ent(policy, seq)
            ratio = torch.exp(new - old)
            clipped = torch.clamp(ratio, 1 - EPS, 1 + EPS) * adv
            ploss = -torch.min(ratio * adv, clipped).mean()
            kl = torch.exp(rl - new) - (rl - new) - 1
            loss = ploss + BETA * kl.mean()
            opt.zero_grad()
            loss.backward()
            gn = nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            clipfracs.append((((ratio < 1 - EPS) | (ratio > 1 + EPS)).float().mean()).item())
            gnorms.append(float(gn))

        with torch.no_grad():
            new, ent = logps_ent(policy, seq)
            klf = (torch.exp(rl - new) - (rl - new) - 1).mean().item()
        hist.append(dict(step=step, reward=rewards.mean().item(), kl=klf,
                         clipfrac=sum(clipfracs) / len(clipfracs), entropy=ent.mean().item(),
                         gnorm=gnorms[0], zero_adv=1.0 if rstd < 1e-6 else 0.0))
    return hist


def last5(h, k):
    return sum(d[k] for d in h[-5:]) / 5


def instability(h):
    return statistics.pstdev([d["reward"] for d in h[-10:]])


def sweep(name, key, vals, **kw):
    print("\n" + "=" * 72)
    print(f"SWEEP: {name}")
    print("=" * 72)
    for v in vals:
        hs = [run_grpo(**{**kw, key: v, "seed": s}) for s in (0, 1)]     # avg over 2 seeds
        fr = sum(last5(h, "reward") for h in hs) / 2
        fk = sum(last5(h, "kl") for h in hs) / 2
        fc = sum(last5(h, "clipfrac") for h in hs) / 2
        inst = sum(instability(h) for h in hs) / 2
        print(f"  {key}={str(v):>6} | final reward {fr:4.2f}/{GEN_LEN} | final KL {fk:6.3f} "
              f"| clipfrac {fc:4.2f} | instability {inst:4.2f}")


def main():
    print("=" * 72)
    print("HEALTHY BASELINE  (lr=3e-3, GROUP=8, EPOCHS=4, EPS=0.2, BETA=0.02)")
    print("=" * 72)
    h = run_grpo()
    print(f"{'step':>4} {'reward':>7} {'kl':>7} {'clipfrac':>9} {'entropy':>8} {'gnorm':>7} {'0adv':>5}")
    for d in h:
        if d["step"] % 5 == 0 or d["step"] == len(h) - 1:
            print(f"{d['step']:>4} {d['reward']:>7.2f} {d['kl']:>7.4f} {d['clipfrac']:>9.2f} "
                  f"{d['entropy']:>8.3f} {d['gnorm']:>7.2f} {d['zero_adv']:>5.0f}")

    sweep("Learning rate", "lr", [1e-3, 3e-3, 1e-2, 3e-2])
    sweep("KL coefficient BETA", "BETA", [0.0, 0.02, 0.1, 0.5])
    sweep("Group size", "GROUP", [2, 4, 8, 16])
    sweep("Epochs (batch reuse)", "EPOCHS", [1, 2, 4, 8])
    sweep("Clip range EPS", "EPS", [0.05, 0.1, 0.2, 0.4])


if __name__ == "__main__":
    main()
