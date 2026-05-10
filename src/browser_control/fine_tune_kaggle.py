import os
import sys

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig
from trl import GRPOTrainer
import wandb

from .browsergym_client import BrowserGymAction
from .browsergym_client import BrowserGymEnv
from .config import FineTuningConfig
from .paths import get_path_model_checkpoints


def rollout_func(
    prompts: list[str],
    trainer: GRPOTrainer,
    client: BrowserGymEnv,
    system_prompt: str,
    max_steps: int,
) -> dict[str, list]:
    episode_prompt_ids: list[list[int]] = []
    episode_completion_ids: list[list[int]] = []
    episode_logprobs: list[list[float]] = []
    completion_rewards: list[float] = []

    print(f"\n[DEBUG] rollout_func called with {len(prompts)} prompts")

    for i, prompt_text in enumerate(prompts):
        print(f"[DEBUG] Processing prompt {i + 1}/{len(prompts)}")
        episode = rollout_once(
            trainer=trainer,
            env=client,
            tokenizer=trainer.processing_class,
            system_prompt=system_prompt,
            dataset_prompt=prompt_text,
            max_steps=max_steps,
        )
        episode_prompt_ids.append(episode["prompt_ids"])
        episode_completion_ids.append(episode["completion_ids"])
        episode_logprobs.append(episode["logprobs"])
        completion_rewards.append(episode["completion_reward"])

    return {
        "prompt_ids": episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "logprobs": episode_logprobs,
        "completion_reward": completion_rewards,
    }


def rollout_once(
    trainer: GRPOTrainer,
    env: BrowserGymEnv,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    dataset_prompt: str,
    max_steps: int,
) -> dict[str, list]:
    from trl.experimental.openenv import generate_rollout_completions

    result = env.reset()
    observation = result.observation

    print("Goal: ", observation.goal)
    print("axtree_txt: ", observation.axtree_txt)

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    step_rewards: list[float] = []
    completion_rewards: list[float] = []

    for step_num in range(max_steps):
        if result.done:
            break

        goal = observation.goal or dataset_prompt
        axtree = observation.axtree_txt or ""
        error = observation.error if observation.last_action_error else ""

        user_prompt = make_user_prompt(goal, step_num, axtree, error)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout_outputs["prompt_ids"])
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"],
            skip_special_tokens=True,
        )

        action_str = parse_action(completion_text)
        print(f"Step {step_num + 1}: {action_str}")

        result = env.step(BrowserGymAction(action_str=action_str))
        observation = result.observation

        step_reward = float(result.reward or 0.0)
        step_rewards.append(step_reward)

        if result.done and step_reward > 0:
            completion_rewards.append(1.0)
        elif result.done and step_reward == 0:
            completion_rewards.append(0.0)
        else:
            completion_rewards.append(step_reward)

    final_reward = completion_rewards[-1] if completion_rewards else 0.0

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "step_rewards": step_rewards,
        "completion_reward": final_reward,
    }


def make_user_prompt(goal: str, step_num: int, axtree: str, error: str = "") -> str:
    prompt_parts = [f"Step {step_num + 1}"]

    if goal:
        prompt_parts.append(f"Goal: {goal}")

    if error:
        prompt_parts.append(f"Previous action error: {error}")

    if axtree:
        max_len = 2000
        axtree_truncated = axtree[:max_len] + "..." if len(axtree) > max_len else axtree
        prompt_parts.append(f"Page structure:\n{axtree_truncated}")

    prompt_parts.append("What action do you take?")
    return "\n\n".join(prompt_parts)


def parse_action(response_text: str) -> str:
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if "(" in line and ")" in line:
            return line
    return "noop()"


def reward_completion(completions: list[str], **kwargs) -> list[float]:
    rewards = kwargs.get("completion_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def create_peft_config(config: FineTuningConfig) -> LoraConfig | None:
    if not config.use_peft:
        return None

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
        use_rslora=config.use_rslora,
    )


def fine_tune_impl(config: FineTuningConfig) -> None:
    os.environ.setdefault("MODEL_CHECKPOINT_ROOT", "/kaggle/working/model_checkpoints")
    os.environ.setdefault("HF_HOME", "/kaggle/working/hf_cache")

    if config.wandb_enabled:
        wandb.init(
            project=config.wandb_project_name,
            name=config.wandb_experiment_name,
            config=config.__dict__,
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"

    print(f"Initializing BrowserGym client at {config.browsergym_url}")
    client = BrowserGymEnv(base_url=config.browsergym_url)

    dataset = Dataset.from_dict({"prompt": [config.default_goal] * config.dataset_size})
    output_dir = get_path_model_checkpoints(config.wandb_experiment_name)

    grpo_config = GRPOConfig(
        max_steps=config.dataset_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        num_generations=config.num_generations,
        generation_batch_size=config.generation_batch_size,
        max_completion_length=config.max_completion_length,
        use_vllm=config.use_vllm,
        vllm_mode=config.vllm_mode,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        output_dir=output_dir,
        logging_steps=config.logging_steps,
        report_to="wandb" if config.wandb_enabled else "none",
    )

    peft_config = create_peft_config(config)

    trainer = GRPOTrainer(
        model=config.model_name,
        reward_funcs=[reward_completion],
        train_dataset=dataset,
        args=grpo_config,
        peft_config=peft_config,
        rollout_func=lambda prompts, trainer: rollout_func(
            prompts=prompts,
            trainer=trainer,
            client=client,
            system_prompt=config.system_prompt,
            max_steps=config.max_steps,
        ),
    )

    trainer.train()

    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)

    if config.push_to_hf:
        trainer.push_to_hub()

    client.close()


def main(config_file_name: str = "kaggle_debug.yaml") -> None:
    config = FineTuningConfig.from_yaml(file_name=config_file_name)
    fine_tune_impl(config)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "kaggle_debug.yaml")
