# Browser Control RL on MiniWoB

This repo is the working training project for fine-tuning small open language models to perform browser actions on MiniWoB-style tasks using GRPO and BrowserGym.

The current setup is:

- **Environment**: BrowserGym served through OpenEnv on Hugging Face Spaces
- **Training**: Kaggle notebooks with 2x T4 GPUs
- **RL algorithm**: TRL `GRPOTrainer`
- **Current task**: `click-test`
- **Current model focus**: `google/gemma-3-270m-it` and `Qwen/Qwen2.5-0.5B-Instruct`

## Current Status

- The BrowserGym v2 Space is stable for repeated MiniWoB sessions.
- Kaggle can connect to the Space and complete two consecutive `click-test` runs successfully.
- A GRPO smoke run completed end-to-end and saved a checkpoint.
- The training code now includes:
  - strict action parsing,
  - rollout JSONL logging,
  - shaped reward for sparse MiniWoB tasks,
  - W&B grouping/tagging support.

## Project Files

- [plan.md](/D:/ML/RL/browser-control/plan.md)
  Accuracy roadmap, bottlenecks, model roadmap, and go/no-go gates.

- [experiments.md](/D:/ML/RL/browser-control/experiments.md)
  Experiment tracker, current queue, run IDs, and config progression.

- [GRPO_TRAINING_FLOW.md](/D:/ML/RL/browser-control/GRPO_TRAINING_FLOW.md)
  Low-level explanation of rollout collection, reward handling, and trainer flow.

## Why RL for Browser Control

Browser control tasks often have multiple valid trajectories. A model can click the correct element immediately, take a longer path, or recover from a mistake. That makes pure supervised data collection expensive and incomplete.

RL is a better fit when:

- the environment can verify success,
- the task has sparse but reliable reward,
- there are many acceptable action sequences,
- the model needs to recover from its own mistakes.

MiniWoB is a good first training target because tasks are narrow, fast to reset, and easy to debug.

## Training Stack

### 1. Environment: BrowserGym via OpenEnv

The environment is BrowserGym running remotely through an OpenEnv-compatible server.

Main benchmark used here:

- [MiniWoB++](https://miniwob.farama.org/)

The current Hugging Face Space serves `click-test` and exposes a generic OpenEnv websocket interface used by the training code.

### 2. Algorithm: GRPO

The trainer uses TRL's `GRPOTrainer`.

At a high level:

1. reset BrowserGym
2. build an action prompt from the current goal and accessibility tree
3. sample model completion(s)
4. parse a single browser action
5. step the environment
6. convert the result into a scalar reward
7. use group-relative reward differences to update the policy

For the current MiniWoB work, sparse terminal reward alone was too weak, so the training code now adds small shaping terms for:

- valid action syntax,
- referencing an element id that exists in the page tree,
- penalizing invalid/no-op behavior when clickable elements exist,
- penalizing environment action errors.

### 3. Policy Models

Primary models:

- `google/gemma-3-270m-it`
- `Qwen/Qwen2.5-0.5B-Instruct`

These are small enough for fast LoRA iterations on Kaggle and large enough to plausibly learn the action grammar.

## Current Training Logic

The main Kaggle entrypoint is:

- [fine_tune_kaggle.py](/D:/ML/RL/browser-control/src/browser_control/fine_tune_kaggle.py)

Important current behaviors:

- the prompt is built from `goal`, `step number`, `axtree_txt`, and previous error
- the parser accepts canonical browser actions only
- rollouts are logged to JSONL for debugging
- reward shaping is configured in YAML
- W&B metadata is configured in YAML

## Current Configs

Primary experiment configs:

- [gemma3_270m_click_shaped.yaml](/D:/ML/RL/browser-control/configs/gemma3_270m_click_shaped.yaml)
- [qwen2_5_0_5b_click_shaped.yaml](/D:/ML/RL/browser-control/configs/qwen2_5_0_5b_click_shaped.yaml)

These are the first real MiniWoB accuracy experiments after the infrastructure smoke test.

## Running on Kaggle

Typical command:

```bash
cd /kaggle/working/miniwob-grpo-finetuning
PYTHONPATH=src .venv/bin/python -m browser_control.fine_tune_kaggle gemma3_270m_click_shaped.yaml
```

For W&B:

1. store `WANDB_API_KEY` in Kaggle Secrets
2. log in inside the notebook before training
3. run the config normally

The experiment configs already include:

- W&B project name
- group name
- job type
- tags
- notes

## What To Look At After Each Run

- TRL metrics:
  - `reward`
  - `reward_std`
  - `rewards/reward_completion/std`
  - `frac_reward_zero_std`
- rollout JSONL logs
- action validity rate
- checkpoint save path

If reward variance stays at zero, do not scale model size or dataset size yet. Fix parser, prompt, or shaping first.

## Near-Term Roadmap

1. Run Gemma 3 270M shaped-reward training.
2. Inspect rollout logs for valid actions and reward variance.
3. Run the Qwen 0.5B comparison.
4. Compare success rate, valid-action rate, and reward variance.
5. Only then consider larger runs or 1B-class models.

## References

- BrowserGym docs: https://browsergym.readthedocs.io/latest/environments/miniwob.html
- MiniWoB docs: https://miniwob.farama.org/
- TRL GRPO docs: https://huggingface.co/docs/trl/grpo_trainer
- Gemma 3 270M IT: https://huggingface.co/google/gemma-3-270m-it
- Qwen2.5 0.5B Instruct: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
