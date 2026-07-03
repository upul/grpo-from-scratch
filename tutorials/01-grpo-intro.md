# Lightweight Guide to understanding GRPO and RL principles

*~11 minute read*

**On this page**

- [Background & Motivation](#background--motivation)
- [The Big Picture: Trial, Error, and a Classroom](#the-big-picture-trial-error-and-a-classroom)
- [Starting Simple: Policy Gradients](#starting-simple-policy-gradients)
- [Problem 1: Reward Is a Noisy, Unfair Teacher](#problem-1-reward-is-a-noisy-unfair-teacher)
- [The Group Baseline: GRPO's One Big Idea](#the-group-baseline-grpos-one-big-idea)
- [Problem 2: Training on Stale Data](#problem-2-training-on-stale-data)
- [Adding Safety Rails: The Clipping Mechanism](#adding-safety-rails-the-clipping-mechanism)
- [Putting It All Together: The Full GRPO Objective](#putting-it-all-together-the-full-grpo-objective)
- [GRPO for LLMs: The Three Models](#grpo-for-llms-the-three-models)
- [Recent Developments: Dr. GRPO and DAPO](#recent-developments-dr-grpo-and-dapo)
- [GRPO vs PPO: What Actually Changed?](#grpo-vs-ppo-what-actually-changed)
- [Conclusion](#conclusion)

## Background & Motivation

This is a mini blog about understanding **Group Relative Policy Optimization (GRPO)**, the reinforcement learning algorithm popularized by DeepSeek and now a default choice for training *reasoning* models — the ones that work through math and code step by step.

If you've read about PPO and kept hearing that *"GRPO is a simpler PPO,"* this guide explains exactly what got simplified and why it was such a good idea. The short version: PPO needs a second neural network — a **critic** — that is large, finicky, and hard to train. GRPO's insight is that for many language tasks you can **throw the critic away** and replace it with something almost embarrassingly simple: just sample a few answers and compare them to each other.

As with the PPO guide, I assume **zero prior RL knowledge**. If you've never touched reinforcement learning, you're in the right place.

---

## The Big Picture: Trial, Error, and a Classroom

GRPO follows the **FAFO principle — Fool Around and Find Out**. The model generates responses, gets scored, and is nudged to produce more of what scored well and less of what scored poorly. That trial-and-error loop is the heart of all the algorithms in this family.

Where PPO hires a personal **coach** (a critic model that predicts how well things will go), GRPO uses the **classroom**. To judge whether an answer is any good, it asks the same question several times, looks at the spread of answers, and grades each one *on the curve*: better than your classmates' average → reinforce it; worse than average → discourage it. No coach needed — the group is its own yardstick.

That's the whole story. Everything below just makes this idea precise, stable, and efficient. We'll build it up one problem at a time.

Here's the high-level loop:

```
1. Roll out: the current model generates a GROUP of answers to each prompt
2. Score:    a reward function/model scores each answer
3. Compare:  each answer's advantage = how it did vs. the group average
4. Update:   nudge the model toward above-average answers (carefully!)
5. Repeat
```

Notice what's *missing* compared to PPO: there's no "train the critic" step. That absence is the entire point of GRPO.

## Starting Simple: Policy Gradients

Let's start with the most basic version of "reward good behavior" and break it as we go.

Our model is a **policy** $\pi_\theta$ — a function with parameters $\theta$ that, given a state, outputs a probability distribution over actions. For a language model:

- a **state** $s_t$ is the prompt plus all tokens generated so far,
- an **action** $a_t$ is the next token,
- the **policy** $\pi_\theta(a_t \mid s_t)$ is just the model's probability for that next token.

The simplest possible idea — the **policy gradient** — is this: if a response got a good reward, increase the probability of the tokens that produced it; if it got a bad reward, decrease them. In one line:

$$\mathcal{L}_{PG} = -\frac{1}{T}\sum_{t=1}^{T} \log \pi_\theta(a_t \mid s_t) \times R$$

where $R$ is the reward for the response and $T$ is the number of tokens. (The minus sign is just because optimizers *minimize*, but we want to *maximize* reward.)

In code:

```python
for response in generated_responses:
    reward = score(response)
    for token in response:
        log_prob = model.log_prob(token)
        token_loss = -log_prob * reward    # push prob up if reward > 0
    update(model)
```

This works in toy settings, but it has a serious flaw — the same one PPO has to fix, and the one that motivates GRPO's whole design. Let's look at it.

## Problem 1: Reward Is a Noisy, Unfair Teacher

Multiplying by the raw reward $R$ has two flaws.

**Flaw A — It has no sense of "expected."** Look closely at what the naive loss actually *does*: it pushes each response's probability **up** by an amount proportional to its reward. Since reward is always positive, this means **every response gets pushed up and nothing ever gets pushed down** — the brilliant, the average, and the mediocre are all reinforced, just by different amounts. So on a hard prompt where the typical score is 8/10, a new response scoring 8 gets boosted enthusiastically, even though it's perfectly *average* — exactly what you'd have expected anyway. It's like grading on a curve where simply matching the class average still earns a trophy. We don't want to reward "average"; we want to reward "**better than expected**" — and to actively *discourage* below-average responses, which raw reward can never do.

**Flaw B — Bad credit assignment.** A response gets *one* reward at the end, but it's made of many tokens. The naive loss assigns that same reward to *every* token equally. Ideally you'd know *which* tokens deserved the credit.

The fix for Flaw A is to subtract a **baseline** $b$ — an estimate of the reward we *expected* — and use the difference, called the **advantage**:

$$\hat{A} = R - b$$

- Advantage > 0: better than expected → increase probability
- Advantage < 0: worse than expected → decrease probability
- Advantage ≈ 0: exactly as expected → leave it alone

Now "average" correctly gets pushed nowhere, good responses get reinforced, and bad ones get actively discouraged. The only question is: **where does the baseline come from?**

> **This is the fork in the road.** PPO answers it by *learning* a baseline — it trains a whole critic network to predict the expected reward at every token. That's powerful (it also fixes Flaw B, giving per-token credit) but expensive and fiddly. GRPO answers the same question with a far cheaper trick, and accepts a coarser fix for Flaw B as the price of simplicity.

## The Group Baseline: GRPO's One Big Idea

Here is the entire trick that defines GRPO. To estimate "the expected reward for this prompt," don't train a model to guess it — just **generate a group of answers to the same prompt and use their average as the baseline.**

Concretely, for a prompt, sample a group of $G$ responses (say 8 of them), score each to get rewards $R_1, \dots, R_G$, and set each response's advantage to its reward minus the group mean, divided by the group's spread:

$$\hat{A}_i = \frac{R_i - \text{mean}(R_1, \dots, R_G)}{\text{std}(R_1, \dots, R_G)}$$

That's it. The **mean** is the baseline (the "expected" score for this prompt), so subtracting it gives you "better or worse than your peers." Dividing by the **standard deviation** just rescales things so the signal is comparable across easy and hard prompts. An answer that beat the group gets a positive advantage; one that lagged gets a negative one; a perfectly average one gets ≈ 0.

This is the same "grading on a curve" idea from Flaw A, made literal — the group *is* the curve.

It's worth pausing on how much this buys you for how little:

- **No critic.** PPO's value network is roughly as large as the model itself and notoriously hard to train well. GRPO deletes it entirely. The baseline costs you nothing but a few extra samples.
- **It's naturally suited to reasoning tasks.** For math or code, you often *want* to sample multiple attempts anyway, and you can check each one's correctness automatically. GRPO turns those attempts into each other's baseline for free.

And the honest tradeoff, returning to Flaw B: every token in a response gets the **same** advantage $\hat{A}_i$ — the whole answer is judged as one unit. GRPO does not pinpoint *which* tokens were good. It bets that a clean per-*response* signal is enough, especially when "right answer vs wrong answer" is what you ultimately care about. (This is exactly the per-token credit that PPO's critic provides and GRPO gives up.)

With the baseline settled, the rest of GRPO is shared, almost verbatim, with PPO. Two more pieces and we're done.

## Problem 2: Training on Stale Data

Generating responses from an LLM is expensive (a full forward pass per token). So we don't want to generate a fresh batch for every gradient step — we want to **reuse** each batch of generations for several updates.

But there's a catch. We generate the group **once** with the model as it is now, then take **multiple** gradient steps on that same batch. By the later steps, the model has changed — yet we're still training on responses the *earlier* model produced. The data no longer matches the current policy. This mismatch is called **off-policy** training.

> It's like practicing your jump shot from a video of yourself last week. You've improved since then, so the footage no longer matches your current form.

The fix is **importance sampling**. We store the log-probabilities from the policy that *actually generated* the data (call it $\pi_{\theta_{old}}$), and correct each update with the ratio:

$$r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,\lt t})}{\pi_{\theta_{old}}(o_{i,t} \mid q, o_{i, \lt t})}$$

- Ratio > 1: the current model likes this token *more* than the generating model did.
- Ratio < 1: it likes it *less*.
- Ratio = 1: they agree; nothing to correct.

The surrogate objective becomes `ratio × advantage` instead of `log_prob × advantage`. If you've seen this in PPO, it's identical here.

## Adding Safety Rails: The Clipping Mechanism

There's a danger with that ratio. If the current model swings far from the generating model (ratio = 5, or 0.01), a single update can be wildly large and destabilize training.

GRPO borrows PPO's fix exactly: **clip** the ratio to a band $[1-\varepsilon, 1+\varepsilon]$ (typically $\varepsilon = 0.2$, so $[0.8, 1.2]$) and take the **minimum** of the clipped and unclipped objectives:

$$\min\Big(r_{i,t}\,\hat{A}_i,\ \text{clip}(r_{i,t},\, 1-\varepsilon,\, 1+\varepsilon)\,\hat{A}_i\Big)$$

The `min` makes the objective **pessimistic**: it removes any incentive to push the ratio outside the band. If an answer was good (positive advantage), the reward for cranking its tokens' probability up is capped once the ratio passes $1+\varepsilon$; if it was bad, the same logic caps how far one step is rewarded for driving probability down. Either way, no single update moves the policy too aggressively — keeping changes *small and safe*.

```python
ratio = torch.exp(current_log_probs - old_log_probs)   # = pi_theta / pi_old
clipped_ratio = torch.clamp(ratio, 1 - eps, 1 + eps)   # eps = 0.2

loss_unclipped = ratio * advantages
loss_clipped = clipped_ratio * advantages
policy_loss = -torch.min(loss_unclipped, loss_clipped).mean()
```

## Putting It All Together: The Full GRPO Objective

Combine the group-relative advantage, importance sampling, and clipping, and add one regularizer — a **KL penalty** that keeps the policy from drifting too far from a frozen **reference model** (the model you started from). The full objective:

$$\mathcal{J}_{GRPO} = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big[\min\big(r_{i,t}\hat{A}_i,\ \text{clip}(r_{i,t}, 1-\varepsilon, 1+\varepsilon)\hat{A}_i\big) - \beta\, D_{KL}\big(\pi_\theta \,\|\, \pi_{ref}\big)\Big]$$

Three things to notice:

**1. The advantage $\hat{A}_i$ is the group-relative one** from earlier — and it's the *same* for every token $t$ in response $i$ (that's the per-response, not per-token, signal).

**2. The KL penalty is a separate term.** This is a subtle but real difference from PPO. In PPO's usual RLHF setup, the KL penalty is folded *into the per-token reward*. GRPO instead keeps it as its own term in the loss, typically using a low-variance, always-non-negative estimator (the "$k_3$" estimator): for log-ratio $\rho = \log\frac{\pi_{ref}}{\pi_\theta}$, use $D_{KL} \approx e^{\rho} - \rho - 1$. The KL anchors the model to sensible language and prevents *reward hacking* (finding degenerate text that fools the scorer).

**3. There is no value-function loss.** PPO's objective has a third term for training the critic. GRPO simply doesn't — there's no critic to train. The objective is shorter because the algorithm is smaller.

The training loop in pseudocode:

```python
for prompt in dataset:
    # 1. Roll out a GROUP from the current policy
    responses = [policy.generate(prompt) for _ in range(G)]
    old_log_probs = policy.log_probs(responses)      # freeze for importance sampling
    rewards = [reward_fn(r) for r in responses]

    # 2. Group-relative advantage (one number per response)
    advantages = (rewards - mean(rewards)) / (std(rewards) + 1e-4)

    # 3. Reuse the batch for several epochs
    for epoch in range(grpo_epochs):
        ratio = torch.exp(policy.log_probs(responses) - old_log_probs)
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        kl = kl_penalty(policy, ref_model, responses)   # k3 estimator
        loss = policy_loss + beta * kl
        loss.backward(); optimizer.step()
```

## GRPO for LLMs: The Three Models

When PPO is used for RLHF, you juggle **four** models: the policy, the critic, a reward model, and a reference model. GRPO drops one of them:

| Model | Role | Trainable? |
| --- | --- | --- |
| **Policy** | The LLM we're improving; generates the group of answers | ✅ yes |
| **Reward model / verifier** | Scores how good each answer is | ❄️ frozen |
| **Reference model** | A frozen copy of the start model; the KL anchor | ❄️ frozen |

The missing fourth is the **critic** — and that's the whole savings. Removing a model the size of the policy itself is a large cut in memory and a large cut in things that can go wrong. In reasoning setups, the "reward model" is often not even a neural network but a simple **verifier** — a function that checks whether the final answer is correct, or whether the code passes its tests. When the reward is a checkable fact, you don't need to *learn* it at all.

## Recent Developments: Dr. GRPO and DAPO

GRPO is young and actively being refined. Two 2025 follow-ups are worth knowing, because they fix biases that vanilla GRPO quietly carries.

**Dr. GRPO** points out that two of GRPO's normalizations introduce subtle biases. Dividing the per-token loss by each response's length ($\frac{1}{|o_i|}$) creates a *length bias* — it ends up nudging the model toward longer wrong answers over time. And dividing the advantage by the group's standard deviation creates a *difficulty bias* — prompts where the model is very consistent (very easy or very hard) get their advantages inflated. The fix is to drop both normalizations and use the plain mean-centered advantage, $\hat{A}_i = R_i - \text{mean}(R)$.

**DAPO** is a recipe of practical fixes for training at scale, including: *Clip-Higher* (an asymmetric clip range with a larger upper bound, which preserves exploration and fights entropy collapse); *dynamic sampling* (throw out prompts where every answer in the group scored the same — those produce a zero advantage and teach nothing); and a *token-level* loss aggregation rather than per-response averaging.

You don't need any of these to get started — vanilla GRPO works — but they're the difference between a toy run and a production one. (Both are covered in more depth in the companion debugging guide.)

## GRPO vs PPO: What Actually Changed?

The cleanest way to hold these two in your head: **GRPO is PPO with the critic removed and replaced by a cheaper baseline.** Everything else — importance sampling, the clipped surrogate, reusing batches, the KL leash — is shared.

| | GRPO | PPO |
| --- | --- | --- |
| **Baseline for advantage** | Mean reward of a sampled group | A learned value function (critic) |
| **Advantage** | One number per *response* | One number per *token*, via GAE |
| **Models in memory** | 3 (no critic) | 4 (policy, critic, reward, reference) |
| **KL penalty** | Separate term in the loss | Folded into the per-token reward |
| **Pros** | Far less memory; simpler; no critic to tune or break | Fine-grained per-token credit; lower-variance on long horizons |
| **Cons** | Coarser, sequence-level signal; needs a group per prompt | The critic doubles memory and is hard to train well |

The deeper point is *why* the trade makes sense for language. Training PPO's critic to predict accurate per-token values is genuinely hard, and it doubles your footprint. When you can cheaply sample a *group* of answers per prompt — which is natural for reasoning tasks with checkable answers — the group mean turns out to be a perfectly good baseline at a fraction of the cost. GRPO bets that for these tasks, the simple empirical baseline beats the expensive learned one in practice. That bet is why it now powers a generation of reasoning models.

## Conclusion

GRPO is one of those rare ideas that's easier than what it replaces. Stripped down, it's three honest moves:

1. **Don't reward "good," reward "better than expected"** — subtract a baseline to get the advantage.
2. **Get that baseline for free from a group** — sample several answers and compare them to each other, instead of training a whole critic to guess the baseline. *This is the move that makes GRPO GRPO.*
3. **Reuse expensive data safely** — importance sampling to correct for staleness, clipping to keep each step small, a KL penalty to stay anchored.

If you internalize those three, you understand not just GRPO but the whole family it belongs to. PPO, RLOO, GRPO, and friends mostly differ in *how they get the baseline*; the clipping and importance sampling at the core stay the same. GRPO's answer just happens to be the simplest one that works — which, in machine learning, is usually the one that wins.

I hope you found this helpful.
