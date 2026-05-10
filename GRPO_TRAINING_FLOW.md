# BrowserGym GRPO Training Flow

This project fine-tunes a language model to control a browser by generating BrowserGym action strings such as `click('13')` or `noop()`.

## Components

The system has three runtime pieces:

- BrowserGym environment server: runs in Hugging Face Spaces at `https://krish-ckpt-browsergym-v2.hf.space`.
- Policy model: the language model being trained, for example `LiquidAI/LFM2-350M`.
- GRPO trainer: TRL's `GRPOTrainer`, running on Kaggle or Modal.

The BrowserGym server owns the browser. Training code never runs Playwright directly. It connects to the server through OpenEnv WebSocket APIs.

## Policy

The policy is the causal language model passed to `GRPOTrainer`.

For every browser step, the code builds a prompt containing:

- the system instruction,
- the task goal,
- the current step number,
- any previous action error,
- the accessibility tree text from BrowserGym.

The model generates text. The parser keeps the first line that looks like a function call. If the model does not produce one, the fallback action is `noop()`.

Example model output:

```text
click('13')
```

That string is sent to BrowserGym as:

```python
BrowserGymAction(action_str="click('13')")
```

## Rollout

A rollout is one attempted browser episode.

The code does this:

1. Calls `env.reset()` to start a fresh MiniWoB task.
2. Reads `observation.goal` and `observation.axtree_txt`.
3. Builds a text prompt for the model.
4. Uses TRL's `generate_rollout_completions(...)` to sample an action from the current policy.
5. Sends the action to BrowserGym with `env.step(...)`.
6. Records token ids, logprobs, and reward.
7. Repeats until BrowserGym says `done=True` or `max_steps` is reached.

The rollout returns:

```python
{
    "prompt_ids": ...,
    "completion_ids": ...,
    "logprobs": ...,
    "step_rewards": ...,
    "completion_reward": ...,
}
```

`completion_reward` is the scalar reward used by GRPO.

## Reward

The reward comes from BrowserGym.

For the `click-test` task:

- clicking the correct button returns a positive reward,
- finishing successfully gives `done=True`,
- wrong/no-op actions usually return `0.0`.

The project applies simple shaping:

```python
if result.done and step_reward > 0:
    completion_reward = 1.0
elif result.done and step_reward == 0:
    completion_reward = 0.0
else:
    completion_reward = step_reward
```

So successful task completion is the main learning signal.

## GRPO

GRPO samples multiple completions for the same prompt and compares their rewards within the group.

In this project:

- `num_generations` controls how many completions/actions are sampled per prompt group.
- `generation_batch_size` controls generation batching.
- `per_device_train_batch_size` controls training batch size.
- `max_completion_length` caps action-generation length.

Better rollouts receive positive relative advantage. Worse rollouts receive lower or negative relative advantage. The trainer updates the policy so future samples are more likely to produce high-reward actions.

GRPO does not train a separate value model, which keeps memory use lower than PPO-style methods.

## LoRA

The Kaggle config uses PEFT/LoRA.

Instead of updating all model weights, training inserts small trainable adapter matrices into target modules such as:

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

This reduces GPU memory and checkpoint size. The saved result is usually adapter weights, not a full standalone model.

## Kaggle Flow

Kaggle runs:

- the policy model,
- vLLM generation,
- GRPO optimization,
- the OpenEnv client.

The HF Space runs:

- BrowserGym,
- MiniWoB static file server,
- Playwright/browser runtime,
- OpenEnv server.

The Kaggle smoke-test entrypoint is:

```bash
PYTHONPATH=src .venv/bin/python -m browser_control.fine_tune_kaggle
```

The default config is:

```text
configs/kaggle_debug.yaml
```

Start with this tiny config first. Increase `dataset_size`, `num_generations`, and `max_steps` only after one full training run completes.

## Known Limits

Kaggle cannot run Docker containers inside the notebook, so BrowserGym should stay in HF Spaces.

Two T4 GPUs will not automatically be used by this code. The initial setup should be treated as a single-GPU run. Multi-GPU training requires a separate distributed setup.

vLLM on T4 can be memory-sensitive. If it fails, reduce:

```yaml
num_generations: 2
generation_batch_size: 2
max_completion_length: 16
vllm_gpu_memory_utilization: 0.05
```

If vLLM still fails, test whether TRL's GRPO path can run with `use_vllm: false` for this setup.
