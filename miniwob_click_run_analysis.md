# MiniWoB Click-Test Run Analysis

Date: 2026-05-10

## Observed Training Pattern

From the Gemma 3 270M run:

- `rewards/reward_completion/mean = -0.15`
- `rewards/reward_completion/std = 0.0`
- `reward_std = 0.0`
- `frac_reward_zero_std = 1.0`
- `loss = 0.0`
- `grad_norm = 0.0`
- `completions/clipped_ratio = 1.0`
- `completions/mean_length = 12.0`

This means every completion in each GRPO group received the same reward, so there was no within-group advantage signal and no effective policy update.

## Root Cause

The rollout logs showed that the model was continuing the prompt text instead of emitting an executable BrowserGym action.

Example:

- prompt ends with `What action do you take?`
- raw completion begins with `Page structure:` and copies the AX tree

The parser therefore fell back to:

- `parsed_action = noop()`
- `valid_action = false`

Under the current reward shaping this produces:

- `invalid_action_penalty = -0.05`
- `noop_with_clickables_penalty = -0.10`
- total shaped reward = `-0.15`

That exactly matches the observed run metrics.

## Why This Happened

### 1. Prompt formatting regressed

The one-step reward rewrite removed tokenizer chat-template formatting and replaced it with a raw concatenated text prompt. For instruction-tuned chat models like `google/gemma-3-270m-it`, that makes the model behave more like a text continuer than an action selector.

### 2. The parser was too strict

The previous parser only accepted lines that exactly matched:

- `click(...)`
- `noop()`
- `fill(...)`
- `keyboard_type(...)`
- `keyboard_press(...)`

Outputs like:

- `Action: click('13')`
- `click(13).`
- `` `click('13')` ``

were treated as invalid even when the action intent was correct.

### 3. Completion length was too short

`max_completion_length = 12` caused completions to be clipped before the model could recover into a clean action line.

## Correct Interpretation of `Action: click('13')`

That output should be treated as a valid action after canonicalization.

Recommended normalization:

- `Action: click('13')` -> `click('13')`
- `click(13)` -> `click('13')`
- `` `click('13')` `` -> `click('13')`

Rewarding should follow the executable action semantics, not strict string identity.

## Changes Applied For Next Experiments

### Prompt path

- Restored tokenizer chat-template formatting when building the training prompt.

### Parser path

- Relaxed action extraction to find the first valid action substring inside the model output.

### Config path

- Increase `max_completion_length` for Gemma and Qwen comparison runs.

## Next Experiment Goals

We want to see these changes in the next run:

- `reward_completion/std > 0`
- `frac_reward_zero_std < 1.0`
- rollout logs containing canonicalized `click('13')` actions
- lower `clipped_ratio`

If these do not improve, the next likely issue is model quality rather than prompt/parser plumbing.

## Follow-Up Gemma Run

After restoring chat-template prompting, relaxing action extraction, and increasing
`max_completion_length` to `24`, the next Gemma 3 270M run changed substantially.

Observed metrics:

- first logged batch:
  - `reward_completion/mean = 0.5`
  - `reward_completion/std = 0.6949`
  - `grad_norm = 1.95`
  - `loss = -0.2165`
- later batches frequently converged to:
  - `reward_completion/mean = 1.15`
  - `reward_completion/std = 0.0`
  - `grad_norm = 0.0`
  - `loss = 0.0`
- `completions/clipped_ratio = 1.0` remained true throughout
- `entropy` collapsed from moderate values to values near `0.003`

## What This Means

This run shows that the pipeline is no longer broken at the reward boundary.

The model is now:

- producing parseable actions often enough to get positive reward
- receiving non-zero gradients early in training
- then collapsing quickly to a nearly deterministic action policy

So the system moved from "no learning signal" to "weak learning signal that collapses too quickly".

## Current Weaknesses

### 1. The prompt is still static across the full dataset

The training dataset is built from one initial environment reset and repeated for all
examples. That means the model is effectively training on one repeated prompt form
with one repeated goal/AX-tree pair.

This makes policy collapse much easier because there is almost no state diversity.

### 2. One-step reward is too easy to exploit

The current reward function scores each completion by:

1. resetting a fresh click-test episode
2. parsing one action
3. stepping once
4. shaping reward from that single step

For click-test, a model can quickly learn a very narrow local policy that often gets
high reward on the repeated prompt. Once all completions in a GRPO group become the same,
reward variance goes to zero and updates stop.

### 3. Reward shaping is still too dense relative to task diversity

The current shaped reward:

- rewards valid format
- rewards referenced element existence
- rewards environment success

On a repeated single-button environment, this can produce rapid convergence to a single
high-reward response without building robustness.

### 4. Completion clipping is still high

`clipped_ratio = 1.0` means the model is still not terminating naturally. Even though
the parser can now recover actions, the generation behavior is still not aligned with
"emit one short action and stop".

### 5. Throughput is poor

The run took about 52 minutes for 32 trainer steps. That is expected because every reward
evaluation does a remote `reset()` and `step()` against the HF Space. This is workable
for debugging but not a strong long-run training setup.

## Should We Rethink The Pipeline?

Yes, partially.

We do **not** need to throw away the whole project. The current pipeline has now proven:

- HF BrowserGym infra works
- reward evaluation works
- Gemma can receive usable reward
- GRPO can update initially

But the current training formulation is too limited for meaningful scaling.

What needs rethinking is not the whole system, but these specific design choices:

### Keep

- HF BrowserGym Space
- remote env scoring
- shaped reward
- W&B tracking
- LoRA / small-model comparison workflow

### Rethink

- repeated static prompt dataset
- one-step-only reward design as the main training target
- exact same click-test state over and over
- lack of explicit stop behavior for generated actions

## Recommended Next Direction

### Short-term next experiments

1. Compare Gemma 270M vs Qwen 0.5B under the same fixed parser/prompt path.
2. Log the first 20 rollout records from each run and compare:
   - raw completion
   - parsed action
   - shaped reward
   - environment reward
3. Add explicit stop-token / shorter stop-aware prompting if supported cleanly.

### Medium-term pipeline improvements

1. Build the dataset from many fresh environment resets instead of one repeated prompt.
2. Add prompt diversity by collecting multiple click-test states and possibly more MiniWoB tasks.
3. Separate "format learning" from "task learning":
   - supervised or constrained action-format warmup
   - then GRPO on environment reward
4. Move from one-step training to bounded multi-step episodes once the single-step policy is stable.

## Bottom Line

The pipeline is no longer failing for plumbing reasons.

The current limitation is now algorithmic and data-design related:

- too little state diversity
- too easy to collapse
- too slow for large-scale exploration

That means the next step is not blind hyperparameter tuning. The next step is improving
the training objective and prompt/state distribution.

## Qwen 0.5B Comparison Run

We then ran the same click-test setup with `Qwen/Qwen2.5-0.5B-Instruct`.

Observed metrics from the initial logged batches:

- `completions/mean_length = 6.0`
- `completions/clipped_ratio = 0.0`
- `completions/mean_terminated_length = 6.0`
- `rewards/reward_completion/mean = 1.15`
- `rewards/reward_completion/std = 0.0`
- `frac_reward_zero_std = 1.0`
- `loss = 0.0`
- `grad_norm = 0.0`
- `entropy ~= 0.018` early and remained very low

## What Qwen Changes And What It Does Not

Compared to Gemma, Qwen immediately showed better output formatting behavior:

- short completions
- natural termination
- no completion clipping

This strongly suggests Qwen is a better base model for canonical action-format learning.

However, Qwen also exposed the same deeper limitation in the current training design:

- all completions in each GRPO group quickly converge to the same high-reward action
- reward variance becomes zero immediately
- gradients vanish
- GRPO stops updating

So Qwen improves the "formatting layer" but does **not** solve the current objective-design problem.

## Updated Interpretation

At this point the experiments support a more precise diagnosis:

### What is solved

- remote BrowserGym infrastructure works
- prompt formatting and parser plumbing are no longer the main blocker
- both Gemma and Qwen can emit executable actions

### What is not solved

- the current dataset is still one repeated state
- the reward is still one-step and too easy to exploit
- the policy collapses before meaningful generalization can happen

## Updated Bottom Line

The current click-test setup is still useful, but only as:

- an infrastructure test
- an action-format bootstrap task
- a narrow smoke test for reward plumbing

It is **not** a sufficient main training setup for RL.

The evidence now points toward a staged pipeline:

1. supervised fine-tuning on diverse MiniWoB states/actions
2. then RL / GRPO on environment reward
3. then broader task generalization and multi-step training

This is now a stronger conclusion than after the earlier Gemma-only runs, because Qwen
showed the same collapse even with much cleaner generation behavior.
