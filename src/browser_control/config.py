from typing import Any, Optional

from datetime import datetime
from pathlib import Path
from typing import Self

import yaml
from pydantic import model_validator
from pydantic_settings import BaseSettings

from .paths import get_path_to_configs


class FineTuningConfig(BaseSettings):
    seed: int
    resume_from_checkpoint: Optional[str] = None # Resume training from this checkpoint (if provided)

    # Language Model parameters
    model_name: str         # HF model name
    max_seq_length: int     # Number of tokens in context window for the LM
    system_prompt: str      # System prompt to provide general instruction to the LM

    # BrowserGym environment
    browsergym_url: str     # URL where the BrowserGym server is accessible
    dataset_size: int       # Number of scenarios we use from the BrowserGymEnvironment
    default_goal: str       # Default goal in case the environment does not provide more explicit instructions
    
    # How long and how aggressive do we train?
    learning_rate: float    # Max learning rate for the optimizer
    warmup_steps: int       # Number of steps to linearly increase learning rate at the start of training

    # vLLM inference
    max_steps: int                      # Max steps per rollout
    per_device_train_batch_size: int    # Number of samples per device per step
    num_generations: int                # Number of completions to generate per prompt
    generation_batch_size: int          # Batch size used during generation (must be divisible by num_generations)
    max_completion_length: int          # Maximum length of generated completions
    use_vllm: bool                      # Use vLLM engine for fast inference
    vllm_mode: str                      # vLLM mode: "colocate" runs generation on the same GPU as training
    vllm_gpu_memory_utilization: float  # Fraction of GPU memory allocated to vLLM

    # experiment tracking
    wandb_enabled: bool
    wandb_project_name: str
    wandb_entity: Optional[str] = None
    wandb_experiment_name: str | None = None
    wandb_group: Optional[str] = None
    wandb_job_type: str = "train"
    wandb_tags: list[str] = []
    wandb_notes: Optional[str] = None
    logging_steps: int                  # How often do we print out training loss?
    push_to_hf: Optional[bool] = True
    config_file_name: Optional[str] = None

    # LoRA-specific hyperparameters for parameter efficient fine-tuning
    use_peft: bool = False  # Default: disabled for backward compatibility
    lora_r: int = 8  # LoRA rank - memory efficient default
    lora_alpha: int = 16  # LoRA scaling factor (2x rank)
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    use_rslora: bool = False
    lora_target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    # Rollout diagnostics and shaped reward for sparse MiniWoB tasks
    rollout_log_path: Optional[str] = None
    valid_action_reward: float = 0.05
    element_id_reward: float = 0.10
    invalid_action_penalty: float = -0.05
    noop_with_clickables_penalty: float = -0.10
    env_error_penalty: float = -0.10

    # max_steps: int = 10000  # increase!
    # save_steps: int = 1000  # increase!
    # eval_steps: int = 1000  # increase!
    # eval_sample_callback_enabled: bool = False

    @classmethod
    def from_yaml(cls, file_name: str) -> Self:
        """
        Loads configuration from a YAML file located in the configs directory.
        """
        file_path = str(Path(get_path_to_configs()) / file_name)
        print(f"Loading config from {file_path}")
        with open(file_path) as f:
            data = yaml.safe_load(f)

        # print('Loaded config:', data)
        data["config_file_name"] = file_name

        return cls(**data)

    @model_validator(mode="after")
    def set_experiment_name(self):
        if self.wandb_experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_short = self.model_name.split("/")[-1]
            self.wandb_experiment_name = (
                f"{model_short}-browsergym-{timestamp}"
            )

        if self.wandb_group is None:
            model_short = self.model_name.split("/")[-1]
            self.wandb_group = f"{model_short}-miniwob-click"

        return self
    
