import os
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
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
CLICK_ARGS_RE = re.compile(r"""^\s*(?:['"]?(?P<bid>\d+)['"]?|id\s*=\s*['"]?(?P<id_bid>\d+)['"]?)\s*$""")


@dataclass
class ParsedAction:
    action_str: str
    valid: bool
    action_name: str
    referenced_bid: str | None = None


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

        bid_match = CLICK_ARGS_RE.match(args)
        referenced_bid = None
        if bid_match:
            referenced_bid = bid_match.group("bid") or bid_match.group("id_bid")

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


def build_training_prompt(config: FineTuningConfig, goal: str, axtree: str) -> str:
    return "\n\n".join(
        [
            config.system_prompt.strip(),
            make_user_prompt(goal, step_num=0, axtree=axtree, error=""),
        ]
    )


def completion_to_text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content")
        return content if isinstance(content, str) else json.dumps(completion, ensure_ascii=True)
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n".join(parts)
    return str(completion)


def parse_action_legacy(response_text: str) -> str:
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if "(" in line and ")" in line:
            return line
    return "noop()"


def make_reward_completion(
    client: BrowserGymEnv,
    config: FineTuningConfig,
    rollout_log_path: str,
):
    def reward_completion(prompts: list[str], completions: list[object], **kwargs) -> list[float]:
        rewards: list[float] = []

        for index, completion in enumerate(completions):
            # For the current MiniWoB click-test experiment, score one fresh episode per completion.
            reset_result = client.reset()
            reset_observation = reset_result.observation
            goal = reset_observation.goal or config.default_goal
            axtree = reset_observation.axtree_txt or ""
            completion_text = completion_to_text(completion)
            parsed_action = parse_action(completion_text)

            step_result = client.step(BrowserGymAction(action_str=parsed_action.action_str))
            step_observation = step_result.observation

            env_reward = float(step_result.reward or 0.0)
            shaped_reward = compute_shaped_reward(
                env_reward=env_reward,
                parsed_action=parsed_action,
                axtree=axtree,
                last_action_error=step_observation.last_action_error,
                config=config,
            )
            rewards.append(shaped_reward)

            append_rollout_log(
                rollout_log_path,
                {
                    "prompt_index": index,
                    "goal": goal,
                    "prompt": prompts[index] if index < len(prompts) else None,
                    "raw_completion": completion_text,
                    "parsed_action": parsed_action.action_str,
                    "valid_action": parsed_action.valid,
                    "action_name": parsed_action.action_name,
                    "referenced_bid": parsed_action.referenced_bid,
                    "referenced_bid_exists": bid_exists(axtree, parsed_action.referenced_bid),
                    "env_reward": env_reward,
                    "shaped_reward": shaped_reward,
                    "done": bool(step_result.done),
                    "env_error": step_observation.error,
                    "last_action_error": step_observation.last_action_error,
                    "axtree_txt": axtree,
                },
            )

        return rewards

    return reward_completion


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
        wandb_config = dict(config.__dict__)
        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            name=config.wandb_experiment_name,
            group=config.wandb_group,
            job_type=config.wandb_job_type,
            tags=config.wandb_tags,
            notes=config.wandb_notes,
            config=wandb_config,
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"

    print(f"Initializing BrowserGym client at {config.browsergym_url}")
    client = BrowserGymEnv(base_url=config.browsergym_url)

    initial_reset = client.reset()
    initial_observation = initial_reset.observation
    initial_goal = initial_observation.goal or config.default_goal
    initial_axtree = initial_observation.axtree_txt or ""
    print(f"Initial goal: {initial_goal}")
    training_prompt = build_training_prompt(config, initial_goal, initial_axtree)

    dataset = Dataset.from_dict({"prompt": [training_prompt] * config.dataset_size})
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
    reward_completion = make_reward_completion(
        client=client,
        config=config,
        rollout_log_path=rollout_log_path,
    )

    trainer = GRPOTrainer(
        model=config.model_name,
        reward_funcs=[reward_completion],
        train_dataset=dataset,
        args=grpo_config,
        peft_config=peft_config,
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
