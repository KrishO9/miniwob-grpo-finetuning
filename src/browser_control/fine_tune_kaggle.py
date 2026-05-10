import os
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

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


ACTION_RE = re.compile(
    r"^(?P<name>click|noop|fill|keyboard_type|keyboard_press)\((?P<args>.*)\)$"
)
BID_RE = re.compile(r"\[(\d+)\]")
CLICKABLE_RE = re.compile(r"\[(\d+)\]\s+(button|link|input|textbox|combobox)", re.I)
QUOTED_BID_RE = re.compile(r"""['"](?P<bid>\d+)['"]""")


@dataclass
class ParsedAction:
    action_str: str
    valid: bool
    action_name: str
    referenced_bid: str | None = None


def rollout_func(
    prompts: list[str],
    trainer: GRPOTrainer,
    client: BrowserGymEnv,
    config: FineTuningConfig,
    rollout_log_path: str,
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
            config=config,
            dataset_prompt=prompt_text,
            rollout_log_path=rollout_log_path,
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
    config: FineTuningConfig,
    dataset_prompt: str,
    rollout_log_path: str,
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

    for step_num in range(config.max_steps):
        if result.done:
            break

        goal = observation.goal or dataset_prompt
        axtree = observation.axtree_txt or ""
        error = observation.error if observation.last_action_error else ""

        user_prompt = make_user_prompt(goal, step_num, axtree, error)
        messages = [
            {"role": "system", "content": config.system_prompt},
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

        parsed_action = parse_action(completion_text)
        action_str = parsed_action.action_str
        print(f"Step {step_num + 1}: {action_str}")

        result = env.step(BrowserGymAction(action_str=action_str))
        observation = result.observation

        step_reward = float(result.reward or 0.0)
        shaped_reward = compute_shaped_reward(
            env_reward=step_reward,
            parsed_action=parsed_action,
            axtree=axtree,
            last_action_error=observation.last_action_error,
            config=config,
        )
        step_rewards.append(step_reward)

        completion_rewards.append(shaped_reward)
        append_rollout_log(
            rollout_log_path,
            {
                "goal": goal,
                "step_num": step_num + 1,
                "raw_completion": completion_text,
                "parsed_action": parsed_action.action_str,
                "valid_action": parsed_action.valid,
                "action_name": parsed_action.action_name,
                "referenced_bid": parsed_action.referenced_bid,
                "referenced_bid_exists": bid_exists(
                    axtree, parsed_action.referenced_bid
                ),
                "env_reward": step_reward,
                "shaped_reward": shaped_reward,
                "done": bool(result.done),
                "env_error": observation.error,
                "last_action_error": observation.last_action_error,
                "axtree_txt": axtree,
            },
        )

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


def parse_action(response_text: str) -> ParsedAction:
    for raw_line in response_text.strip().split("\n"):
        line = raw_line.strip().strip("`")
        if not line:
            continue
        match = ACTION_RE.match(line)
        if not match:
            continue

        action_name = match.group("name")
        args = match.group("args").strip()
        if action_name == "noop":
            return ParsedAction("noop()", args == "", "noop")

        bid_match = QUOTED_BID_RE.search(args)
        referenced_bid = bid_match.group("bid") if bid_match else None

        if action_name == "click" and referenced_bid:
            return ParsedAction(f"click('{referenced_bid}')", True, "click", referenced_bid)

        if action_name == "fill" and referenced_bid:
            return ParsedAction(line, True, "fill", referenced_bid)

        if action_name in {"keyboard_type", "keyboard_press"} and args:
            return ParsedAction(line, True, action_name)

    return ParsedAction("noop()", False, "noop")


def extract_bids(axtree: str) -> set[str]:
    return set(BID_RE.findall(axtree or ""))


def has_clickables(axtree: str) -> bool:
    return bool(CLICKABLE_RE.search(axtree or ""))


def bid_exists(axtree: str, bid: str | None) -> bool:
    return bid is not None and bid in extract_bids(axtree)


def compute_shaped_reward(
    env_reward: float,
    parsed_action: ParsedAction,
    axtree: str,
    last_action_error: bool,
    config: FineTuningConfig,
) -> float:
    reward = env_reward

    if parsed_action.valid:
        reward += config.valid_action_reward
    else:
        reward += config.invalid_action_penalty

    if bid_exists(axtree, parsed_action.referenced_bid):
        reward += config.element_id_reward

    if parsed_action.action_name == "noop" and has_clickables(axtree):
        reward += config.noop_with_clickables_penalty

    if last_action_error:
        reward += config.env_error_penalty

    return float(reward)


def append_rollout_log(path: str, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def parse_action_legacy(response_text: str) -> str:
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
    rollout_log_path = config.rollout_log_path or str(Path(output_dir) / "rollouts.jsonl")
    print(f"Writing rollout logs to {rollout_log_path}")

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
            config=config,
            rollout_log_path=rollout_log_path,
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
