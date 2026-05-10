# MiniWoB Browser-Control Accuracy Plan

**Current state**: BrowserGym v2 is running on Hugging Face Spaces at `https://krish-ckpt-browsergym-v2.hf.space`. Kaggle can connect to it, run two consecutive `click-test` sessions, receive reward `1.0` for a correct click, and complete a tiny GRPO smoke run with checkpoint save.

**Primary objective**: improve success rate on MiniWoB-style browser tasks by making the policy reliably emit valid BrowserGym actions from accessibility-tree observations.

## Claim Map

| Claim | Why It Matters | Minimum Evidence |
| --- | --- | --- |
| C1: The training loop can learn from environment reward. | Without non-zero reward variance, GRPO has no useful learning signal. | `reward_completion/std > 0`, `frac_reward_zero_std < 1.0`, and some successful rollouts. |
| C2: Better action formatting and reward shaping improve sample efficiency. | MiniWoB rewards are sparse; invalid actions waste rollout budget. | Higher valid-action rate and success rate versus raw sparse reward. |
| C3: Small instruction models can learn simple web actions with LoRA. | Establishes the baseline before scaling to 1B+ models. | Current small model improves on held-out MiniWoB tasks after controlled experiments. |

## Key References

- TRL `GRPOTrainer` supports custom reward functions and forwards extra rollout fields to reward functions. Custom `rollout_func` is responsible for returning the correct number of completions per prompt when using `num_generations`: https://huggingface.co/docs/trl/grpo_trainer
- BrowserGym action primitives include `click`, `fill`, `keyboard_type`, `noop`, scrolling, and other browser actions: https://browsergym.readthedocs.io/latest/core/action_space.html
- BrowserGym MiniWoB integration provides MiniWoB tasks through Gym environments: https://browsergym.readthedocs.io/latest/environments/miniwob.html
- MiniWoB has a lower-level action-space model and supports action-space customization: https://miniwob.farama.org/content/action_space/
- Gemma 3 270M is the primary next model track because it is small enough for fast Kaggle LoRA iterations and belongs to the Gemma 3 family we want to optimize: https://huggingface.co/google/gemma-3-270m-it
- Qwen2.5-0.5B-Instruct is the primary comparison model because it is small, permissively licensed, and strong at structured outputs: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- SmolLM2-1.7B-Instruct is Apache-2.0 and designed for lightweight instruction-following use cases: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct
- Gemma 3 1B is a reasonable later LoRA target, but requires accepting Google's model license before use: https://huggingface.co/google/gemma-3-1b-it

## Current Bottlenecks

1. **Zero reward variance**: the first smoke run finished but logged zero reward and zero loss. That means GRPO had no useful advantage signal.
2. **Loose action parser**: `parse_action()` accepts any line containing `(` and `)`, and otherwise returns `noop()`. This hides model failures.
3. **Sparse terminal reward only**: for a one-click task, the policy must already produce a near-perfect action before it receives positive reward.
4. **No rollout audit file**: we currently print actions, but we do not persist raw completions, parsed actions, valid-action flags, reward, and errors.
5. **Action schema mismatch risk**: BrowserGym supports richer action signatures than our simplified prompt. The simplified schema is good for early MiniWoB, but must be validated against actual accepted actions.
6. **No baseline policy**: we need a deterministic heuristic baseline to prove the environment and reward are correct independent of model training.
7. **Potential `num_generations` mismatch**: TRL expects custom rollout functions to honor generation grouping. We need to verify our rollout function returns the right number of completions per prompt.

## Accuracy Strategy

### Stage 0: Preserve Working Infrastructure

- Keep `use_vllm: false` until the training loop has non-zero reward variance.
- Keep BrowserGym on HF Spaces and Kaggle for model training.
- Keep `dataset_size`, `num_generations`, and `max_steps` small until we can diagnose rollouts.
- Avoid changing model size and reward logic in the same experiment.

### Stage 1: Instrument Rollouts

Add a JSONL rollout log containing:

- `run_id`
- `prompt`
- `goal`
- `axtree_txt`
- `raw_completion`
- `parsed_action`
- `valid_action`
- `referenced_bid_exists`
- `env_error`
- `done`
- `reward`
- `step_num`

Success criterion:

- We can answer why reward is zero in each rollout without reading console output.

### Stage 2: Tighten Action Format

Replace the current parser with a strict parser:

- Accept only supported canonical forms for early MiniWoB:
- `click('13')`
- `noop()`
- `fill('42', 'text')`
- `keyboard_type('text')`
- `keyboard_press('Enter')`

Early simplification:

- For `click-test`, allow only `click(...)` and `noop()`.
- Reject explanations and malformed actions explicitly.
- Add a format reward/penalty so the model learns the grammar before solving the task.

Success criterion:

- Valid-action rate rises above 80% on debug rollouts.

### Stage 3: Add Shaped Rewards

Keep environment success reward as the main signal, but add shaping:

- `+1.0` task success from BrowserGym.
- `+0.05` output is one valid action and nothing else.
- `+0.10` action references an element id present in the accessibility tree.
- `-0.05` invalid format.
- `-0.10` `noop()` when clickable elements exist.
- `-0.10` environment reports `last_action_error`.

Reasoning:

- GRPO needs within-group reward differences. Sparse-only reward is too brittle for early small-model training.

Success criterion:

- `reward_completion/std > 0` in most training steps.
- `frac_reward_zero_std < 1.0`.

### Stage 4: Establish Baselines

Run these before longer training:

- **Heuristic click baseline**: parse the first visible button id from `axtree_txt` and click it.
- **Prompt-only baseline**: frozen model with current prompt and strict parser.
- **Shaped-reward baseline**: same frozen model but with shaped reward logging.

Success criterion:

- Heuristic baseline solves `click-test`.
- Prompt-only model has measurable valid-action rate.
- Shaping separates good and bad outputs.

### Stage 5: Scale Training Carefully

Only after Stages 1-4:

- Increase `dataset_size` to 8, then 32, then 128.
- Increase `num_generations` to 4.
- Keep `max_steps: 2` for `click-test`.
- Keep LoRA enabled.
- Keep `use_vllm: false` until rollouts are stable and slow generation becomes the bottleneck.

Success criterion:

- Training improves success rate over frozen prompt-only baseline on repeated seeds.

### Stage 6: Add Task Diversity

After `click-test` works:

- Add simple click tasks.
- Add simple text-entry tasks.
- Add dropdown/select tasks.
- Keep task families separate at first to avoid noisy diagnosis.

Suggested task progression:

- `click-test`
- simple button-click tasks
- simple form-fill tasks
- simple choose/select tasks
- mixed MiniWoB subset

Success criterion:

- The model generalizes to at least one unseen task from the same action family.

## Model Roadmap

### Primary Track: Gemma 3 270M

- `google/gemma-3-270m-it`: primary improvement target.
- Why this first: low memory, fast LoRA iteration, and aligned with the Gemma family before trying 1B.
- First config: `configs/gemma3_270m_click_shaped.yaml`.
- Gate: if it cannot produce valid action syntax after parser/shaping, do not scale to Gemma 1B yet.

### Primary Comparison: Qwen2.5 0.5B

- `Qwen/Qwen2.5-0.5B-Instruct`: strong candidate for structured action output; Apache-2.0; small enough for Kaggle LoRA.
- First config: `configs/qwen2_5_0_5b_click_shaped.yaml`.
- Why this comparison matters: if Qwen learns the action grammar faster, we can separate model-family weakness from reward/prompt weakness.

### Smoke / Legacy Baselines

- `LiquidAI/LFM2-350M`: current working smoke model; keep as infra regression baseline.
- `google/functiongemma-270m-it`: optional legacy 270M baseline; not the main path unless Gemma 3 270M access fails.

### Later Open Models

- `HuggingFaceTB/SmolLM2-360M-Instruct`: very small baseline if available in current dependencies.
- `HuggingFaceTB/SmolLM2-1.7B-Instruct`: larger but still plausible with LoRA on 2x T4; good later comparison.

### 1B-Class Models

- `google/gemma-3-1b-it`: reasonable after small-model experiments. Use LoRA, no vLLM initially, and confirm HF license access.
- `meta-llama/Llama-3.2-1B-Instruct`: possible later, but license/access and Kaggle HF auth must be handled.

### Recommendation

Do not switch to 1B yet. First make Gemma 3 270M and Qwen2.5 0.5B produce valid actions and non-zero reward variance. Bigger models help only after the reward and parser pipeline are not wasting rollouts.

## Go / No-Go Gates

| Gate | Requirement | If Failed |
| --- | --- | --- |
| G1: Infra | Two consecutive env sessions succeed. | Fix HF Space/env first. |
| G2: Rollout audit | JSONL shows raw completion, parsed action, reward. | Do not run longer training. |
| G3: Valid actions | Valid-action rate > 80%. | Improve prompt/parser/format reward. |
| G4: Reward variance | `reward_completion/std > 0` in most groups. | Add stronger shaping or easier task. |
| G5: Success | At least some task successes before scaling. | Add heuristic warm-start/SFT data. |

## Risks

- **HF Space latency**: process isolation is reliable but slower. Mitigation: keep early experiments small.
- **Sparse reward collapse**: all rewards zero. Mitigation: action-format and element-reference shaping.
- **Parser reward hacking**: model learns valid-looking but useless actions. Mitigation: keep success reward dominant.
- **Task overfitting**: model memorizes `click-test`. Mitigation: introduce held-out MiniWoB tasks after first success.
- **Model mismatch**: some small models may not follow action-only instructions. Mitigation: compare frozen valid-action rates before training.
