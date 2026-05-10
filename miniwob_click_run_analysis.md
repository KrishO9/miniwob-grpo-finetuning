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
