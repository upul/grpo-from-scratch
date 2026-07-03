#!/usr/bin/env python
"""
GRPO from scratch on a toy model -- the whole algorithm in ~80 lines, runnable in
seconds on CPU. A tiny GPT learns to emit vowels; the reward simply counts them.

The point is to see the five GRPO steps with nothing else in the way:
    1. sample a GROUP of completions from the current policy
    2. score them and compute a group-relative advantage
    3. snapshot the generating policy (old_logp) and the frozen reference (ref_logp)
    4. the clipped surrogate loss + KL penalty
    5. reuse the batch for a few epochs

Run:  python grpo_toy.py
Expect the mean reward to climb from ~1-2 toward 6/6 within ~50 steps.
"""
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)

# ----- task: 10-letter vocab, reward = number of vowels (a, e) in the output -----
CHARS = list("abcdefghij")
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB = len(CHARS)
PROMPT = [STOI["a"]]        # every rollout starts from the same 1-token prompt
GEN_LEN = 6                 # completions are 6 tokens long
VOWELS = set("ae")


def reward_fn(tokens):
    return float(sum(1 for t in tokens if ITOS[t] in VOWELS))


# ----- a tiny causal Transformer -----
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
        mask = torch.triu(torch.ones(T, T) * float("-inf"), diagonal=1)
        a, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return self.head(x)


@torch.no_grad()
def generate(model, n):
    """Autoregressively sample n completions of length GEN_LEN from PROMPT."""
    seq = torch.tensor(PROMPT).repeat(n, 1)
    for _ in range(GEN_LEN):
        probs = F.softmax(model(seq)[:, -1, :], dim=-1)
        seq = torch.cat([seq, torch.multinomial(probs, 1)], dim=1)
    return seq


def per_token_logps(model, seq):
    """log pi(token_t) for each generated token -> (n, GEN_LEN)."""
    logp = F.log_softmax(model(seq[:, :-1]), dim=-1)
    tok_logp = logp.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)
    return tok_logp[:, -GEN_LEN:]                 # keep only completion tokens


def main():
    policy = TinyGPT()
    ref = copy.deepcopy(policy)                   # frozen reference for the KL penalty
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)

    GROUP, EPOCHS, EPS, BETA, STEPS = 8, 4, 0.2, 0.02, 60
    hist = []

    for step in range(STEPS):
        # 1. ROLLOUT: sample a group from the current policy
        seq = generate(policy, GROUP)
        comps = seq[:, 1:].tolist()
        rewards = torch.tensor([reward_fn(c) for c in comps])

        # 2. GROUP-RELATIVE ADVANTAGE (the one big idea: the group IS the baseline)
        adv = ((rewards - rewards.mean()) / (rewards.std() + 1e-4)).unsqueeze(1)

        # 3. SNAPSHOT the generating policy + the frozen reference
        with torch.no_grad():
            old_logp = per_token_logps(policy, seq)
            ref_logp = per_token_logps(ref, seq)

        # 4-5. UPDATE: reuse the batch for several epochs
        for _ in range(EPOCHS):
            new_logp = per_token_logps(policy, seq)
            ratio = torch.exp(new_logp - old_logp)
            clipped = torch.clamp(ratio, 1 - EPS, 1 + EPS) * adv
            policy_loss = -torch.min(ratio * adv, clipped).mean()
            kl = torch.exp(ref_logp - new_logp) - (ref_logp - new_logp) - 1   # k3, >= 0
            loss = policy_loss + BETA * kl.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        hist.append(rewards.mean().item())
        if step % 10 == 0 or step == STEPS - 1:
            sample = "".join(ITOS[t] for t in comps[0])
            print(f"step {step:3d} | mean reward {rewards.mean():.2f}/{GEN_LEN} | sample '{sample}'")

    print(f"\nfirst-5-step avg reward: {sum(hist[:5]) / 5:.2f}"
          f"   ->   last-5-step avg reward: {sum(hist[-5:]) / 5:.2f}")


if __name__ == "__main__":
    main()
