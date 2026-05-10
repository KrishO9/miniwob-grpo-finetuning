# Browser Control Setup Guide

This guide covers three pieces:

1. Modal setup for remote GRPO training
2. Hugging Face setup for the BrowserGym Space and optional model uploads
3. Repo setup for `examples/browser-control`

It is written against the current code in this repo, not an idealized version. Where the code has rough edges, they are called out explicitly.

## What This Example Actually Needs

The training stack in this example has three moving parts:

- A remote BrowserGym environment, typically exposed as a Hugging Face Space URL
- A Modal job that runs TRL `GRPOTrainer` plus vLLM on a GPU
- A local checkout of this repo used to launch the Modal job

The code paths to know first:

- Training entrypoint: [src/browser_control/fine_tune.py](./src/browser_control/fine_tune.py)
- Config schema: [src/browser_control/config.py](./src/browser_control/config.py)
- Modal image and secrets: [src/browser_control/modal_infra.py](./src/browser_control/modal_infra.py)
- Training configs: [configs/](./configs)

## Prerequisites

You need:

- Python `3.12`
- `uv`
- A Modal account and CLI login
- A Hugging Face account
- Optional: a Weights & Biases account if you want logging

If you plan to push trained checkpoints to Hugging Face Hub, you also need a Hugging Face access token.

## 1. Modal Setup

Official references:

- Modal CLI tokens: <https://modal.com/docs/reference/cli/token>
- Modal `modal run`: <https://modal.com/docs/reference/cli/run>
- Modal secrets: <https://modal.com/docs/guide/secrets>
- Modal volumes: <https://modal.com/docs/guide/volumes>

### 1.1 Install Modal locally

From inside `examples/browser-control`:

```powershell
uv sync
```

That installs the local Python dependencies declared in [pyproject.toml](./pyproject.toml). The remote Modal container installs its own dependencies separately from [`modal_infra.py`](./src/browser_control/modal_infra.py).

### 1.2 Important OpenEnv dependency note

Do not mix a Git install of `openenv` with a separately pinned `openenv-core` wheel unless you have verified that exact pair works together.

This example is wired to use OpenEnv from one pinned Git commit:

```text
bf5e968286e0d49cdc03fd904d48faff4b15a437
```

If you are building a Hugging Face Space for BrowserGym, keep the OpenEnv install source consistent. In practice that means:

- keep the `openenv` Git install
- remove any extra `openenv-core==...` line from `requirements.txt` or your Dockerfile

If you install both independently, you can hit import errors such as:

```text
ModuleNotFoundError: No module named 'openenv_core.env_server.serialization'
```

### 1.3 Authenticate the Modal CLI

Install the CLI if needed, then authenticate:

```powershell
uv run modal token new
```

You can verify the active credentials with:

```powershell
uv run modal token info
```

### 1.4 Create required Modal secret(s)

The current code only injects one secret into the training container:

- `wandb-secret`

That happens in [`get_secrets()`](./src/browser_control/modal_infra.py). If `wandb_enabled: true` in your YAML config, create the secret before training:

```powershell
uv run modal secret create wandb-secret WANDB_API_KEY=your_wandb_api_key
```

If you do not want WandB yet, set this in your config instead:

```yaml
wandb_enabled: false
```

### 1.5 Modal volumes used by this example

This code uses two named Modal volumes:

- `hf-model-cache`
- `browser-control-fine-tune-with-grpo`

They are created automatically because the code calls `modal.Volume.from_name(..., create_if_missing=True)` in [`modal_infra.py`](./src/browser_control/modal_infra.py).

You do not need to create them manually unless you want to inspect them later with the CLI.

Useful commands:

```powershell
uv run modal volume list
uv run modal volume ls hf-model-cache
uv run modal volume ls browser-control-fine-tune-with-grpo
```

### 1.6 GPU and cost expectations

The training function currently requests an `A100` in [`fine_tune.py`](./src/browser_control/fine_tune.py). That is a high-end GPU choice for a small-model example.

If you want lower cost, change the decorator argument before running:

```python
gpu="A100"
```

to something cheaper that is supported for your model and vLLM setup, for example an `L40S` if you have validated it. The code even has `#"L40S"` commented next to the current value.

### 1.7 Launching a Modal job

The intended command is:

```powershell
uv run modal run -m src.browser_control.fine_tune --config-file-name lfm2_350m.yaml
```

You can also use the Make target:

```powershell
make fine-tune config=lfm2_350m.yaml
```

Important caveat:

- The README says `make run ...`, but the actual Make target is `fine-tune` in [Makefile](./Makefile).

## 2. Hugging Face Setup

Official references:

- Spaces overview: <https://huggingface.co/docs/hub/main/spaces-overview>
- Docker Spaces: <https://huggingface.co/docs/hub/main/spaces-sdks-docker>
- Repositories: <https://huggingface.co/docs/hub/main/repositories-getting-started>

There are two distinct Hugging Face roles in this example:

1. A Hugging Face Space hosts the BrowserGym environment
2. A Hugging Face model repo can store your trained checkpoint or LoRA adapter

### 2.1 Sign in locally

Install the Hub CLI if needed and authenticate:

```powershell
uv run python -m pip install huggingface_hub
hf auth login
```

This is useful locally for testing and for creating repos. It does not automatically authenticate the remote Modal container.

### 2.2 BrowserGym environment via Space

The training code connects to a BrowserGym server at `browsergym_url` from your YAML config.

Examples already in this repo:

- `https://burtenshaw-browsergym-v2.hf.space`
- `https://paulescu-browsergym-book-flight.hf.space`

That URL is passed into:

```python
client = BrowserGymEnv(base_url=config.browsergym_url)
```

in [`fine_tune.py`](./src/browser_control/fine_tune.py).

You have two options.

#### Option A: Reuse an existing Space URL

This is the fastest path if the public Space is available and already serves the task you want.

Use one of the existing config files and keep `browsergym_url` unchanged.

#### Option B: Run your own Hugging Face Space

Use this if:

- you want a private environment
- you want a different BrowserGym task mix
- the public Space is unstable
- you want to control updates and uptime

General Docker Space flow:

1. Create a new Space on Hugging Face
2. Choose `Docker` as the SDK
3. Push the Dockerized BrowserGym/OpenEnv server code
4. Wait for the Space to build
5. Use the resulting `https://<space-name>.hf.space` URL as `browsergym_url`

Space settings that matter:

- Visibility: public, protected, or private
- Hardware tier if your Space needs more than CPU
- Secrets and variables under the Space Settings page

Notes:

- For this example, the environment server can usually run on CPU; the GPU is needed on Modal for training and vLLM.
- If you duplicate or recreate a Space, remember that Hugging Face secrets are not copied automatically into duplicated Spaces. Variables can be copied; secrets are not.

### 2.3 Model repo for pushing checkpoints

If `push_to_hf: true`, `trainer.push_to_hub()` is called at the end of training in [`fine_tune.py`](./src/browser_control/fine_tune.py).

Create a model repository first if you want an explicit destination, or let the trainer create/upload under your configured context if supported.

You can create a repo from CLI or Python. CLI example:

```powershell
hf repo create your-username/lfm2-350m-browsergym-test --type model
```

### 2.4 Important limitation in the current code

The current Modal app only injects `wandb-secret`. It does not inject a Hugging Face token secret into the remote container.

That means `push_to_hf: true` may fail from the Modal job unless you patch the code to provide a token, for example by adding a second Modal secret and injecting it from [`modal_infra.py`](./src/browser_control/modal_infra.py).

As the code stands today, the safest first-run setting is:

```yaml
push_to_hf: false
```

Then verify training works before wiring remote Hub auth.

## 3. Repo Setup

### 3.1 Working directory

Use:

```powershell
cd examples/browser-control
```

This directory contains its own `pyproject.toml`, `Makefile`, `configs/`, and `src/`.

### 3.2 Install local dependencies

```powershell
uv sync
```

This creates the local environment from [pyproject.toml](./pyproject.toml).

### 3.3 Review the default configs

The main config files are:

- [configs/lfm2_350m.yaml](./configs/lfm2_350m.yaml): full fine-tune
- [configs/lfm2_350m_lora.yaml](./configs/lfm2_350m_lora.yaml): LoRA fine-tune
- [configs/functiongemma_270m.yaml](./configs/functiongemma_270m.yaml): alternate small model
- [configs/lfm2_350m_book_flight.yaml](./configs/lfm2_350m_book_flight.yaml): alternate task endpoint

The config fields are validated by [`FineTuningConfig`](./src/browser_control/config.py).

The most important fields are:

- `model_name`
- `browsergym_url`
- `dataset_size`
- `learning_rate`
- `num_generations`
- `generation_batch_size`
- `max_steps`
- `use_peft`
- `push_to_hf`
- `wandb_enabled`

### 3.4 Recommended first-run config

For a cheap, low-risk first run, start from [configs/lfm2_350m_lora.yaml](./configs/lfm2_350m_lora.yaml) and reduce it further:

```yaml
dataset_size: 20
num_generations: 2
generation_batch_size: 2
max_steps: 5
wandb_enabled: false
push_to_hf: false
```

Why:

- LoRA is cheaper than full fine-tuning
- smaller `dataset_size` reduces total update steps
- fewer generations reduce rollout cost
- disabling WandB and HF upload removes external auth failure points

### 3.5 Run training

Preferred direct command:

```powershell
uv run modal run -m src.browser_control.fine_tune --config-file-name lfm2_350m_lora.yaml
```

Equivalent Make target:

```powershell
make fine-tune config=lfm2_350m_lora.yaml
```

### 3.6 Where outputs go

Inside the remote Modal container, checkpoints are saved under:

```text
/model_checkpoints/<experiment-name>
```

That path is built in [`paths.py`](./src/browser_control/paths.py) and backed by the Modal volume named `browser-control-fine-tune-with-grpo`.

### 3.7 Evaluating a trained model

There is an evaluation script at [src/browser_control/evaluate.py](./src/browser_control/evaluate.py), but it has a filename mismatch right now:

- It tries to load `lfm2_350m_debugging.yaml`
- The repo contains `lfm2_350m_debug.yaml`

You should fix that before relying on the evaluation script.

## 4. Running Your Own BrowserGym Space

If you want your own environment service instead of the public Space, the practical workflow is:

1. Build or copy the BrowserGym/OpenEnv server code into a separate repo
2. Create a Hugging Face Docker Space
3. Push the environment server
4. Wait for the Space to become healthy
5. Set `browsergym_url` in your training config to the Space URL
6. Test `env.reset()` and one `env.step(...)` locally before launching a long Modal job

The exact server implementation is not vendored inside `examples/browser-control`; `BrowserGymEnv` is imported from the OpenEnv stack, not implemented locally in this repo.

## 5. Suggested End-to-End First Run

### Minimal first run

1. `cd examples/browser-control`
2. `uv sync`
3. `uv run modal token new`
4. Copy `configs/lfm2_350m_lora.yaml` and make a debug version with:
   - `dataset_size: 20`
   - `num_generations: 2`
   - `generation_batch_size: 2`
   - `max_steps: 5`
   - `wandb_enabled: false`
   - `push_to_hf: false`
5. Confirm the `browsergym_url` points to a live Space
6. Run:

```powershell
uv run modal run -m src.browser_control.fine_tune --config-file-name your_debug_config.yaml
```

### After that works

1. Add WandB if you want experiment tracking
2. Add a Hugging Face token to the Modal runtime if you want remote `push_to_hf`
3. Increase `dataset_size`
4. Increase `num_generations`
5. Try a second model config such as [configs/functiongemma_270m.yaml](./configs/functiongemma_270m.yaml)

## 6. Known Issues In This Example

- README command mismatch: it says `make run`, but the Makefile target is `make fine-tune`
- Evaluation config mismatch: `lfm2_350m_debugging.yaml` is referenced, but only `lfm2_350m_debug.yaml` exists
- `push_to_hf` is optimistic in the current code because no Hugging Face secret is injected into Modal
- `seed`, `max_seq_length`, and `resume_from_checkpoint` exist in config but are not wired into the trainer logic yet
- The code is text-only right now. It uses `observation.axtree_txt`, not screenshots, during training

## 7. External References

- Modal token docs: <https://modal.com/docs/reference/cli/token>
- Modal run docs: <https://modal.com/docs/reference/cli/run>
- Modal secrets docs: <https://modal.com/docs/guide/secrets>
- Modal volumes docs: <https://modal.com/docs/guide/volumes>
- Hugging Face Spaces overview: <https://huggingface.co/docs/hub/main/spaces-overview>
- Hugging Face Docker Spaces: <https://huggingface.co/docs/hub/main/spaces-sdks-docker>
- Hugging Face repositories getting started: <https://huggingface.co/docs/hub/main/repositories-getting-started>
