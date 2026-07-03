---
title: Debugging and Tuning GRPO — A Practical Reference
description: A model-agnostic, reusable field guide for debugging and tuning any GRPO run — the metrics to log, what each one means and how it fails, what every hyperparameter does and which gauge it moves, plus a baseline-calibration workflow and a pre-flight checklist. Includes a verified toy run as evidence for the universal patterns.
---

# Debugging and Tuning GRPO — A Practical Reference

*~15 minute read · keep this open during runs*

This is a standing reference for debugging and tuning **any** GRPO training run, from a 0.5B model on GSM8K to a large reasoning model. The framework here is model-agnostic. To keep it concrete, a tiny verified toy run is used as *evidence* that the patterns are real — but the toy's specific numbers are illustration, not values to copy.

## The one thing to internalize

There are two layers, and mixing them up is the most common mistake:

1. **The framework transfers to every run.** Which metrics to watch, what each one *means*, how it fails, which knob fixes it, and the order to tune in — all identical at any scale.
2. **The healthy *numbers* are per-run calibration.** A "normal" KL, clip fraction, or entropy curve depends entirely on your model, task, reference, and reward. You do not import these; you **establish them with a baseline run** and then react when a gauge leaves its band.

So the skill is *flying by instruments*. This guide gives you the instrument panel and how to read it; each new model/task gives you the gauge readings that count as "normal" for it.

## The instrument panel: what to always log

Log these every run, from step 1. Reward alone is not enough — the loss is meaningless in GRPO (it hovers near zero by design), so these are how you actually see what's happening.

```python
# per update (inside the epoch loop):
clipfrac = (((ratio < 1-EPS) | (ratio > 1+EPS)).float() * completion_mask).sum() / completion_mask.sum()
gnorm    = nn.utils.clip_grad_norm_(policy.parameters(), 1.0)

# per step (after the update):
kl       = ((torch.exp(ref_logp - new_logp) - (ref_logp - new_logp) - 1) * completion_mask).sum() / completion_mask.sum()
entropy  = (token_entropy * completion_mask).sum() / completion_mask.sum()
comp_len = completion_mask.sum(1).float().mean()                 # avg completion length
zero_adv = (group_reward_std < 1e-6).float().mean()              # fraction of dead groups
# and assert the sanity invariant on epoch 0:
assert ((ratio - 1).abs() * completion_mask).max() < 1e-4        # masking is correct
```

## Metric reference card (model-agnostic)

The reusable core. For each metric: what it means, what *healthy* looks like (qualitatively — calibrate the exact band per run), the trouble signature, and the knob that fixes it.

| Metric | Measures | Healthy (any model) | Trouble signature → fix |
|---|---|---|---|
| **Reward** | The actual objective | Rises, then plateaus | Flat at the floor → reward/parser bug, or LR too low. Rises but generations look *worse* → reward hacking → raise `BETA` / stop earlier |
| **KL to reference** | Drift from the start model | Grows slowly, stays bounded | Climbing fast + text degrading → drifting → raise `BETA`. Pinned near 0 + reward stuck → `BETA` too high |
| **Clip fraction** | How often updates hit the clip | Small, nonzero (a few % to ~20%) | Sustained above ~30–35% → steps too big → lower LR or `EPOCHS` |
| **Entropy** | Exploration remaining | Declines gradually, never to ~0 | Crashes toward 0 → entropy collapse → entropy bonus, Clip-Higher, lower LR |
| **Grad norm** | Update violence | Stable; brief spikes OK | Sustained large, or NaN → instability → grad-clip, lower LR, check masking |
| **Zero-advantage rate** | Fraction of groups with no reward spread | Low (some near convergence) | High → prompts too easy/hard for the model → dynamic sampling, curriculum, larger `GROUP` |
| **Completion length** | Avg tokens generated | Stable, or grows for a *good* reason | Ballooning while reward flat → length hacking, or hitting max-tokens with no answer → length penalty, read generations |
| **`ratio == 1` @ epoch 0** | Masking/bookkeeping correctness | Exactly 1.0 | Not 1.0 → masking or old-logprob bug (a *code* bug, not a hyperparameter) |

> The headline habit: **watch relationships, not single numbers.** Reward up + clip fraction falling + entropy alive + grad norm stable = healthy, regardless of the absolute values.

## What each hyperparameter does (and the gauge it moves)

The *directions* below are structural — they hold on any model. The starting ranges are sane defaults; tune from there.

| Knob | Effect (universal direction) | Typical real-LLM range | Gauge to watch |
|---|---|---|---|
| **`lr`** | Too high → instability/oscillation; too low → slow | 1e-6 – 1e-5 | instability, clip fraction, grad norm |
| **`BETA`** (KL) | Too high → suppresses learning, pins KL; too low → drift / reward hacking | 0 – 0.04 | KL-to-reference |
| **`GROUP`** | Smaller → noisier baseline (the group *is* the baseline) | 8 – 16 (never < 4) | reward noise, zero-advantage rate |
| **`EPOCHS`** | More → more off-policy drift per rollout → clip fraction rises | 1 – 4 | clip fraction |
| **`EPS`** (clip) | Sets how much clipping happens; rarely needs tuning | 0.2 (or 0.2/0.28 for Clip-Higher) | clip fraction |

Note on learning rate: real LLMs use *far* smaller LRs than small toys (often ~1e-6). Always recalibrate `lr` to your model; never port it across scales.

## Workflow: establish your baseline, then fly by it

Do this at the start of every new model/task. It's what makes the rest of the guide usable.

1. **Start from safe defaults** (the range table above; `BETA` low, moderate `lr`, `GROUP ≥ 8`, `EPOCHS` small).
2. **Run a short baseline** — a few dozen steps — with the full dashboard logging.
3. **Record *your* healthy bands**: your typical KL, your typical clip fraction, your entropy-decay shape, your normal completion length. These are now your reference points.
4. **Thereafter, react when a gauge leaves its band**, using the reference card and the recipe below. You're no longer guessing what "normal" is — you measured it.

## The tuning recipe (priority order)

Tune in this order; each step names the gauge to watch so you're never guessing.

1. **Get reward to MOVE at all.** Safe defaults, and confirm the `ratio == 1` epoch-0 check passes. *If reward won't move,* it's almost never the hyperparameters — check the reward function/parser first (print `(completion, extracted, gold)` triples), then the zero-advantage rate, then whether `lr` is simply too low.
2. **If unstable** (reward oscillates/diverges): lower `lr`, then lower `EPOCHS`. *Watch clip fraction and instability fall.*
3. **If reward stalls below where it should** (meaningful reference): leash too tight → lower `BETA`. *Watch KL — if suppressed while reward is stuck, that confirms it.*
4. **If reward climbs but quality degrades** (reward hacking): the opposite → raise `BETA` and/or stop earlier. *Watch KL climbing alongside reward.*
5. **Set `GROUP` for signal quality** — as large as your generation budget allows; 8–16 typical, never below 4.
6. **Leave `EPS` at 0.2** unless deliberately tuning exploration (then Clip-Higher).

## Pre-flight checklist (before you burn compute)

GRPO runs are slow and expensive to fail. Front-load these cheap checks every time:

- [ ] `ratio == 1` on epoch 0 (asserts masking + bookkeeping are correct).
- [ ] Reward function returns sane values on a few hand-checked examples — *read the triples*.
- [ ] Advantages are not all zero (groups have reward spread).
- [ ] Completion mask isolates only completion tokens (not prompt, not padding).
- [ ] The setup can overfit a tiny batch (if it can't win on one, it won't on a million).
- [ ] All dashboard metrics are logging from step 1.

## What to expect differently on a real LLM vs. a toy

Because the toy converges in seconds, its dashboard is unrealistically *clean*. Real runs are messier and slower — here's the recalibration, metric by metric, so the clean toy curves don't set false expectations:

| Metric | Toy behavior (illustration) | Expect on a real LLM |
|---|---|---|
| Reward | climbs to the ceiling in ~10 steps | slow, partial climb; may plateau well below "solved"; verifiable reward is often a noisy 0/1 fraction |
| KL | runs into the 1–3 range (reference = *random* init) | much smaller (reference = SFT model); judge the *trend*, not the magnitude |
| Clip fraction | falls to 0 at convergence | stays low-to-moderate; rarely hits 0 (the policy never fully "converges") |
| Entropy | dips then recovers cleanly | gradual decline; main job is to prevent a *collapse* to 0 |
| Zero-adv rate | flips to 1 once the task is solved | persistent and partial; high *early* means difficulty mismatch, not "solved" |
| Length | fixed tiny horizon | hundreds of tokens; watch for length creep and max-token cutoffs with no answer |

The framework is identical; the *texture* is different. Plan for plateaus, partial credit, and parser-induced zero rewards that masquerade as "not learning."

## Evidence: the patterns on a verified toy run

Everything above is grounded in a real instrumented run of the from-scratch toy (tiny model; "generate 8 tokens, reward = count of `a`/`e`"). The numbers are toy-specific; the **shapes** are what generalize.

**A healthy run** — note clip fraction falling, entropy recovering, zero-adv flipping once solved:

```
step  reward    kl  clipfrac  entropy  gnorm  0adv
   0    1.12  0.13     0.45     2.08   1.04    0
   5    4.50  1.04     0.18     1.41   0.96    0
  10    7.88  2.17     0.02     0.27   0.76    0
  15    8.00  1.16     0.00     0.06   0.00    1
  49    8.00  0.88     0.00     0.63   0.03    1
```

**Hyperparameter sweeps** (final reward over 8, averaged across 2 seeds) — each confirms a universal direction:

```
BETA (KL):   0.0 → 8.00 | 0.02 → 7.95 | 0.1 → 6.41 | 0.5 → 3.09 (KL pinned at 0.31)
             ↳ too-high BETA suppresses learning and pins KL — the clearest knob.
lr:        1e-3 → 7.99 | 3e-3 → 7.95 | 1e-2 → 7.96 | 3e-2 → 7.69 (instability 6× higher)
             ↳ too-high LR still learns but oscillates.
GROUP:        2 → 6.40 |    4 → 7.95 |    8 → 7.95 |   16 → 7.99 (smoothest)
             ↳ small group = noisy baseline; diminishing returns past 8.
EPOCHS:       1 → 8.00 |    2 → 8.00 |    4 → 7.95 |    8 → 7.85 (clipfrac 0.00 → 0.06)
             ↳ more reuse = more off-policy drift, visible as rising clip fraction.
EPS:       0.05 → 8.00 |  0.1 → 7.78 |  0.2 → 7.95 |  0.4 → 7.96 (least sensitive)
             ↳ 0.2 is the leave-it-alone default.
```

Read those as *directions you can rely on*, then calibrate the actual magnitudes on your own model with a baseline run.

## Conclusion

Debugging and tuning GRPO stops being guesswork once you can read the dashboard: reward says *whether* it works, clip fraction and instability say if steps are too big, KL says if the leash is wrong, entropy says if exploration is alive, and the zero-advantage rate says if your groups are teaching anything. Each hyperparameter moves one specific gauge — so tuning is "read the gauge, turn the matching knob." That mapping is permanent and works on any model. What changes per run is only what "normal" looks like — which you establish with a baseline run, not by importing anyone's numbers, including this guide's.

Keep this open during your next run.
