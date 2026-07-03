# Implementing GRPO from Scratch on a Tiny LLM

*~12 minute read*

**On this page**

- [Background & Motivation](#background--motivation)
- [GRPO in One Loop](#grpo-in-one-loop)
- [The Setup: A Tiny LLM and a Verifiable Reward](#the-setup-a-tiny-llm-and-a-verifiable-reward)
- [Step 1: Sample a Group (Rollout)](#step-1-sample-a-group-rollout)
- [Step 2: Score and Compute Group-Relative Advantages](#step-2-score-and-compute-group-relative-advantages)
- [Step 3: Per-Token Log-Probs (and the Masking Trap)](#step-3-per-token-log-probs-and-the-masking-trap)
- [Step 4: The Clipped Surrogate Loss + KL Penalty](#step-4-the-clipped-surrogate-loss--kl-penalty)
- [Step 5: Reuse the Batch (the Off-Policy Epochs)](#step-5-reuse-the-batch-the-off-policy-epochs)
- [The Full Training Loop](#the-full-training-loop)
- [Watching It Learn](#watching-it-learn)
- [From Toy to a Real LLM](#from-toy-to-a-real-llm)
- [Conclusion](#conclusion)

## Background & Motivation

The GRPO tutorial explained the *ideas*. This one builds the *thing*. By the end you'll have a complete, working GRPO trainer in about 100 lines of plain PyTorch — no TRL, no `verl`, no magic — and you'll have watched it actually move a language model's behavior.

To keep every moving part visible, we'll train a **deliberately tiny** GPT on a **verifiable** task, so we never need a separate reward model and can confirm with our own eyes that learning happened. The exact same loop scales to a real model like Qwen or Llama; at the end I'll show precisely which lines change. Everything below was run start to finish, and I'll show the real output.

The beauty of GRPO is that it's genuinely simple to implement. Because it **has no critic** (that's its whole selling point over PPO), there's no value network to build, no GAE recursion to get right — just sample a group, see which responses did better than their peers, and push toward those.

## GRPO in One Loop

Here's the entire algorithm in pseudocode. Five steps, repeated:

```
repeat:
  1. ROLLOUT   sample G responses to a prompt from the current policy
  2. SCORE     reward each response; advantage = how it compares to the group
  3. SNAPSHOT  record the log-probs of the policy that generated them
  4. UPDATE    for a few epochs: clipped surrogate loss + KL-to-reference penalty
  5. repeat
```

That's it. Let's build each piece, then assemble the whole.

## The Setup: A Tiny LLM and a Verifiable Reward

Our model is a small GPT over a 10-letter alphabet (`a`–`j`). Our task is intentionally trivial so the reward needs no neural network: **generate strings with as many vowels as possible** (our only vowels are `a` and `e`). The reward is just a count.

Why this task? Because it's *verifiable* — we compute the reward with one line of Python — and because it's the same shape as real GRPO tasks like math or code, where a grader checks the answer. The model starts out babbling random letters; if GRPO works, it should learn to emit mostly `a`s and `e`s.

```python
import torch, torch.nn as nn, torch.nn.functional as F, copy, random
torch.manual_seed(0); random.seed(0)

chars = list("abcdefghij")
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
VOCAB = len(chars)
VOWELS = set("ae")
PROMPT  = [stoi["a"]]   # every rollout starts from this 1-token prompt
GEN_LEN = 6             # generate 6 completion tokens

def reward_fn(tokens):                  # tokens: list of completion ids
    return float(sum(1 for t in tokens if itos[t] in VOWELS))
```

The model itself is a minimal GPT: a token embedding, a position embedding, one causal self-attention block, an MLP, and an output head. The details aren't the point — any autoregressive model with a `forward` that returns per-position logits will do.

```python
class TinyGPT(nn.Module):
    def __init__(self, vocab=VOCAB, d=32, heads=2, ctx=16):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.head = nn.Linear(d, vocab)

    def forward(self, idx):                       # (B, T) -> (B, T, vocab)
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T))
        h = self.ln1(x)
        mask = torch.triu(torch.ones(T, T) * float("-inf"), diagonal=1)
        a, _ = self.attn(h, h, h, attn_mask=mask)  # causal attention
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return self.head(x)
```

Crucially, GRPO needs **two** copies of the model: the **policy** we train, and a frozen **reference** that anchors the KL penalty so the policy can't drift into nonsense.

```python
policy = TinyGPT()
ref = copy.deepcopy(policy)                 # frozen reference (the KL anchor)
for p in ref.parameters(): p.requires_grad_(False)
opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
```

## Step 1: Sample a Group (Rollout)

The "group" in *Group* Relative Policy Optimization is the heart of the method. Instead of training a critic to predict expected reward, GRPO just samples **several** responses to the same prompt and lets them serve as each other's baseline.

We generate autoregressively, **sampling** (not greedy `argmax`) — sampling is what gives us the diversity GRPO needs to compare responses against each other.

```python
@torch.no_grad()
def generate(model, n):
    """Sample n completions from the same prompt."""
    seq = torch.tensor(PROMPT).repeat(n, 1)        # (n, 1)
    for _ in range(GEN_LEN):
        logits = model(seq)[:, -1, :]              # next-token logits
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)          # SAMPLE, for diversity
        seq = torch.cat([seq, nxt], dim=1)
    return seq                                      # (n, 1+GEN_LEN)
```

## Step 2: Score and Compute Group-Relative Advantages

Now the one equation that *is* GRPO. We score each of the $G$ responses, then turn raw rewards into **advantages** by asking "how did this response do *relative to its group*?" — subtract the group mean, divide by the group std:

$$\hat{A}_i = \frac{R_i - \text{mean}(R_{1..G})}{\text{std}(R_{1..G}) + \epsilon}$$

A response that beat its peers gets a positive advantage (do more of this); one that lagged gets a negative advantage (do less). The group mean *is* the baseline — the exact role PPO's value function plays, accomplished for free by sampling.

```python
seq = generate(policy, GROUP)
comps = seq[:, 1:].tolist()
rewards = torch.tensor([reward_fn(c) for c in comps])

# group-relative advantage — the heart of GRPO
adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
adv = adv.unsqueeze(1)                              # (GROUP, 1), broadcasts over tokens
```

Note every token in a response shares that response's single advantage — GRPO's signal is per-*response*, not per-token. (The `+ 1e-4` matters: if all responses score the same, the std is zero and you'd divide by zero. We covered that "zero-advantage trap" in the debugging guide.)

## Step 3: Per-Token Log-Probs (and the Masking Trap)

To compute the policy-gradient update we need $\log \pi(a_t \mid s_t)$ for each **generated** token. A model `forward` gives logits at every position; we take the log-softmax and pick out the probability the model assigned to the token that was actually produced.

The trap that bites *everyone* the first time: you must include **only the completion tokens** in the loss, never the prompt. Train on the prompt tokens and your gradients are meaningless. Here we slice off the last `GEN_LEN` positions to keep exactly the generated tokens.

```python
def per_token_logps(model, seq):
    """log pi(token_t | token_<t) for each GENERATED token -> (n, GEN_LEN)."""
    logits = model(seq[:, :-1])                    # predict positions 1..T-1
    logp = F.log_softmax(logits, dim=-1)
    targets = seq[:, 1:]                            # the actual next tokens
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return tok_logp[:, -GEN_LEN:]                  # keep ONLY completion tokens
```

## Step 4: The Clipped Surrogate Loss + KL Penalty

Two pieces here, both inherited conceptually from PPO.

**The clipped surrogate.** Because we'll reuse each batch for several gradient steps (Step 5), the policy drifts away from the one that generated the data. We correct for that with the importance ratio $r_{i,t} = \frac{\pi_\theta}{\pi_{old}}$ and **clip** it to $[1-\varepsilon, 1+\varepsilon]$ so no single update is too violent:

$$\mathcal{L}^{CLIP} = -\frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t \min\!\Big(r_{i,t}\hat{A}_i,\ \text{clip}(r_{i,t}, 1-\varepsilon, 1+\varepsilon)\hat{A}_i\Big)$$

**The KL penalty.** We add a penalty for drifting from the frozen reference, keeping outputs sane. We use the **$k_3$ estimator** — it's always non-negative and low-variance. For log-ratio $r = \log\pi_{ref} - \log\pi_\theta$:

$$\text{KL} \approx e^{r} - r - 1$$

```python
new_logp = per_token_logps(policy, seq)
ratio = torch.exp(new_logp - old_logp)             # pi_theta / pi_old
unclipped = ratio * adv
clipped = torch.clamp(ratio, 1 - EPS, 1 + EPS) * adv
policy_loss = -torch.min(unclipped, clipped).mean()

# per-token KL to the frozen reference (k3 estimator: always >= 0)
kl = torch.exp(ref_logp - new_logp) - (ref_logp - new_logp) - 1
loss = policy_loss + BETA * kl.mean()
```

## Step 5: Reuse the Batch (the Off-Policy Epochs)

Generating from an LLM is the expensive part, so we don't throw a batch away after one gradient step — we take several. This is why we **snapshot** the generating policy's log-probs (`old_logp`) *before* the update loop: every epoch compares the current policy against that fixed snapshot via the ratio. The clipping from Step 4 is exactly what makes this safe.

```python
with torch.no_grad():
    old_logp = per_token_logps(policy, seq)        # frozen: the generating policy
    ref_logp = per_token_logps(ref, seq)           # frozen: the KL anchor

for _ in range(EPOCHS):                            # reuse the same batch
    # ... Step 4 loss, then:
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)   # always grad-clip
    opt.step()
```

## The Full Training Loop

Here is everything assembled — a complete, runnable GRPO trainer. The hyperparameters at the top are the only knobs.

```python
GROUP  = 8      # responses sampled per prompt (the "group")
EPOCHS = 4      # gradient steps reusing each batch of generations
EPS    = 0.2    # clip range
BETA   = 0.02   # KL penalty coefficient
STEPS  = 60

for step in range(STEPS):
    # 1. ROLLOUT: sample a group from the CURRENT policy
    seq = generate(policy, GROUP)
    comps = seq[:, 1:].tolist()
    rewards = torch.tensor([reward_fn(c) for c in comps])

    # 2. GROUP-RELATIVE ADVANTAGE
    adv = ((rewards - rewards.mean()) / (rewards.std() + 1e-4)).unsqueeze(1)

    # 3. SNAPSHOT the generating policy + reference (frozen)
    with torch.no_grad():
        old_logp = per_token_logps(policy, seq)
        ref_logp = per_token_logps(ref, seq)

    # 4-5. UPDATE: reuse the batch for several epochs
    for _ in range(EPOCHS):
        new_logp = per_token_logps(policy, seq)
        ratio = torch.exp(new_logp - old_logp)
        clipped = torch.clamp(ratio, 1 - EPS, 1 + EPS) * adv
        policy_loss = -torch.min(ratio * adv, clipped).mean()
        kl = torch.exp(ref_logp - new_logp) - (ref_logp - new_logp) - 1
        loss = policy_loss + BETA * kl.mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

    if step % 10 == 0 or step == STEPS - 1:
        sample = "".join(itos[t] for t in comps[0])
        print(f"step {step:3d} | mean reward {rewards.mean():.2f}/{GEN_LEN} | sample '{sample}'")
```

**Two frozen things — don't confuse them.** Step 3 computes *two* sets of "old" log-probs, and they play different roles. `old_logp` is a snapshot of the **generating policy**, taken fresh **every step** right after sampling — it's what the importance ratio $\pi_\theta/\pi_{old}$ corrects against during the reuse epochs. `ref_logp` comes from `ref`, a copy of the **starting model frozen once before training** and never updated — it's the anchor for the KL penalty.

| | `old_logp` (generating policy) | `ref` (reference model) |
|---|---|---|
| **Purpose** | Correct for stale data *within* the reuse epochs | Keep the policy near the *original* model |
| **What it is** | The policy at generation time | A frozen copy of the starting model |
| **Refreshed?** | Every step (inside the loop) | Never — frozen once, globally |

A natural question is whether `ref` should be re-frozen each step too. It should not: if `ref` tracked the current policy, the KL term would collapse to $D_{KL}(\pi_\theta \| \pi_\theta) \approx 0$ and stop anchoring anything. The memory hook: **`old_logp` exists because of *epoch reuse*; `ref` exists because of *long-run drift*.**

## Watching It Learn

Running the script above produces this (your exact numbers will vary with hardware/seed, but the trend is the point):

```
step   0 | mean reward 1.62/6 | sample 'jbfahe'
step  10 | mean reward 5.50/6 | sample 'eafaea'
step  20 | mean reward 5.88/6 | sample 'aaaeaa'
step  30 | mean reward 5.88/6 | sample 'ajaaaa'
step  40 | mean reward 5.88/6 | sample 'eaeaee'
step  50 | mean reward 6.00/6 | sample 'eaeaee'
step  59 | mean reward 6.00/6 | sample 'eeaaae'
```

It works. The model starts at ~1.6 vowels out of 6 (random — about what you'd expect from 2 vowels in a 10-letter alphabet) and climbs to a perfect 6/6, with samples visibly shifting from gibberish like `jbfahe` to all-vowel strings like `eeaaae`. No critic, no reward model, ~100 lines. That is the whole appeal of GRPO.

A nice thing to watch alongside reward is the **clip fraction** (the share of tokens hitting the clip bound). Early on it's high — the policy is moving fast — and as the model converges and updates shrink, it falls toward zero. That's the clipping doing its job: loud when needed, silent once you've arrived.

## From Toy to a Real LLM

The loop you just built is the *real* algorithm. Going from this toy to fine-tuning an actual model is mostly swapping components, not rewriting logic:

- **Swap the model.** Replace `TinyGPT` with a pretrained model, e.g. `AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")`. Your `policy`, frozen `ref`, `generate`, and `per_token_logps` all keep the same shape — a HF model's logits are the same `(B, T, vocab)` you already handle.
- **Swap the reward** for a real verifiable grader: does the generated answer match the gold solution (math), pass unit tests (code), or follow the required format? GRPO shines exactly when rewards are *checkable*, which is why it took over reasoning-model training.
- **Use real prompts.** Replace the single fixed prompt with a dataset, and mask each example's prompt tokens out of the loss (variable prompt lengths make the masking from Step 3 essential rather than a convenience).
- **Batch the generation** and use a fast inference path (e.g. vLLM) — rollout is the bottleneck at scale.
- **Consider the refinements** from the debugging guide once you're running: Dr. GRPO's removal of the std and length normalizations, and DAPO's Clip-Higher and dynamic sampling. None change the skeleton you built here; they're small, targeted edits to the advantage and the loss aggregation.

The conceptual core — sample a group, score it, advantage = relative-to-group, clipped update with a KL leash — does not change between this 10-letter toy and a frontier reasoning model. You've now written all of it.

## Conclusion

GRPO is one of those rare algorithms that's easier to implement than to describe. Stripped down, it's a sampling loop wrapped around a single idea: *let a handful of attempts at the same problem grade each other, and lean toward whatever beat the average.* Removing PPO's critic doesn't just save memory — it removes the hardest-to-debug component entirely, which is why a complete, correct implementation fits on one screen.

If you typed this in and watched your own reward curve climb, you now understand GRPO more concretely than any equation can convey. From here, swapping in a real model and a real reward is a change of degree, not of kind.
