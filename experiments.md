# Browser-Control Experiment Tracker

**Environment**: `https://krish-ckpt-browsergym-v2.hf.space`

**Known working state**:

- HF BrowserGym v2 rebuild includes worker-process isolation for BrowserGym episodes.
- Kaggle can run two consecutive `click-test` sessions successfully.
- First GRPO smoke run completed and saved a checkpoint.
- Current smoke run was done with `use_vllm: false`.

## Completed Runs

| Run ID | Date | Purpose | Config / Model | Result | Notes |
| --- | --- | --- | --- | --- | --- |
| R000 | 2026-05-10 | HF env repeated-session test | `click-test`, manual test client | PASS | Two consecutive `/ws` sessions completed; correct click returned `reward: 1.0`. |
| R001 | 2026-05-10 | GRPO smoke test | `configs/kaggle_debug.yaml`, `LiquidAI/LFM2-350M`, `use_vllm: false` | PASS | Training finished `2/2` steps and saved checkpoint. Reward/loss were zero. |

## Immediate Run Queue

| Run ID | Milestone | Purpose | Variant | Metrics | Priority | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R002 | M0 | Add rollout audit logging | Current model, no training change | Raw completion, parsed action, reward, env error | MUST | TODO | Required before accuracy work. |
| R003 | M0 | Frozen prompt-only evaluation | Current prompt/parser, 20 episodes | success rate, valid-action rate, noop rate | MUST | TODO | Establish baseline before changing rewards. |
| R004 | M0 | Heuristic click baseline | Parse first button id and click | success rate | MUST | TODO | Confirms env/reward independent of LLM. |
| R005 | M1 | Strict parser only | Regex parser, no reward shaping | valid-action rate, success rate | MUST | TODO | Measures parser/prompt impact. |
| R006 | M1 | Format reward shaping | Strict parser + format reward | reward std, valid-action rate | MUST | TODO | Goal: non-zero GRPO reward variance. |
| R007 | M1 | Element-id reward shaping | Add `bid exists` reward | reward std, success rate | MUST | TODO | Goal: distinguish valid click targets. |
| R008 | M2 | First full Gemma 3 270M shaped run | `configs/gemma3_270m_click_shaped.yaml` | reward mean/std, valid-action rate, checkpoint | MUST | READY | First real learning test after smoke. |
| R009 | M2 | Repeat Gemma 3 270M shaped run | Same as R008, different seed | stability across seed | SHOULD | TODO | Avoid over-reading one run. |
| R010 | M3 | Qwen2.5 0.5B shaped comparison | `configs/qwen2_5_0_5b_click_shaped.yaml` | reward mean/std, valid-action rate, checkpoint | MUST | READY | Compare structured-output behavior. |
| R011 | M3 | Larger Gemma 3 270M run | `dataset_size: 128`, same shaping | success rate trend | SHOULD | TODO | Only if R008 has non-zero variance. |

## Later Model Comparison Queue

| Run ID | Milestone | Purpose | Model | Config | Priority | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R020 | M4 | Legacy smoke regression | `LiquidAI/LFM2-350M` | LoRA, no vLLM | SHOULD | TODO | Keep as infra regression baseline. |
| R021 | M4 | Optional FunctionGemma baseline | `google/functiongemma-270m-it` | LoRA, no vLLM | NICE | TODO | Only if Gemma 3 270M access fails or we need 270M family comparison. |
| R022 | M4 | Lightweight 1.7B candidate | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | LoRA, no vLLM | NICE | TODO | Check memory and speed. |
| R023 | M5 | 1B Gemma candidate | `google/gemma-3-1b-it` | LoRA, no vLLM | NICE | TODO | Requires HF license access. |

## Suggested Config Progression

### First Full Gemma 3 270M Run

```yaml
model_name: google/gemma-3-270m-it
dataset_size: 32
per_device_train_batch_size: 1
num_generations: 4
generation_batch_size: 4
max_steps: 2
max_completion_length: 24
use_vllm: false
learning_rate: 1.0e-4
```

Command:

```bash
PYTHONPATH=src .venv/bin/python -m browser_control.fine_tune_kaggle gemma3_270m_click_shaped.yaml
```

### Qwen2.5 0.5B Comparison Run

```yaml
model_name: Qwen/Qwen2.5-0.5B-Instruct
dataset_size: 32
per_device_train_batch_size: 1
num_generations: 4
generation_batch_size: 4
max_steps: 2
max_completion_length: 24
use_vllm: false
learning_rate: 1.0e-4
```

Command:

```bash
PYTHONPATH=src .venv/bin/python -m browser_control.fine_tune_kaggle qwen2_5_0_5b_click_shaped.yaml
```

### Later 1B LoRA Trial

```yaml
model_name: google/gemma-3-1b-it
dataset_size: 8
per_device_train_batch_size: 1
num_generations: 2
generation_batch_size: 2
max_steps: 2
max_completion_length: 24
use_vllm: false
use_peft: true
lora_r: 8
lora_alpha: 16
```

## Metrics To Track Every Run

- `success_rate`: percentage of episodes ending with `done=True` and positive reward.
- `valid_action_rate`: percentage of generations parsed as supported actions.
- `noop_rate`: percentage of parsed actions equal to `noop()`.
- `env_error_rate`: percentage of steps with `last_action_error=True`.
- `reward_mean`: mean rollout reward.
- `reward_std`: reward standard deviation within GRPO groups.
- `frac_reward_zero_std`: from TRL logs; should decrease below `1.0`.
- `mean_completion_length`: useful for detecting rambly completions.
- `checkpoint_path`: saved model directory.

## Run Decision Rules

- If `valid_action_rate < 50%`, do not increase dataset size. Improve prompt/parser first.
- If `reward_std == 0` for most groups, do not run longer. Add or adjust shaping.
- If heuristic baseline fails, debug BrowserGym/action format before model training.
- If HF Space latency dominates, keep `dataset_size <= 32` until reward signal is proven.
- If a model cannot follow one-line action format when frozen, deprioritize it.

## Open Implementation Tasks

- [x] Add rollout JSONL logging.
- [x] Add strict action parser.
- [x] Add valid-action and element-id detectors.
- [x] Add shaped reward fields to rollout outputs.
- [ ] Add frozen evaluation script.
- [ ] Add heuristic click baseline script.
- [x] Add configs for R008 and R010.
- [x] Add model configs for Gemma 3 270M and Qwen2.5 0.5B.
- [ ] Add model config for Gemma 3 1B after small-model signal is confirmed.
