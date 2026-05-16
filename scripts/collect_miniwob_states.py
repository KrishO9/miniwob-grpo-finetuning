from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from browser_control.browsergym_client import BrowserGymEnv
from browser_control.data.jsonl import append_jsonl
from browser_control.data.schemas import RawStateRecord


def parse_seed_range(value: str) -> list[int]:
    if ":" in value:
        start_raw, end_raw = value.split(":", 1)
        start = int(start_raw)
        end = int(end_raw)
        if end <= start:
            raise ValueError("Seed range end must be greater than start")
        return list(range(start, end))
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_bucket(path: str | Path, bucket_name: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        buckets = yaml.safe_load(f) or {}
    if bucket_name not in buckets:
        known = ", ".join(sorted(buckets.keys()))
        raise KeyError(f"Unknown bucket '{bucket_name}'. Known buckets: {known}")
    bucket = buckets[bucket_name]
    if not bucket.get("tasks"):
        raise ValueError(f"Bucket '{bucket_name}' has no tasks")
    return bucket


def collect_states(
    *,
    browsergym_url: str,
    bucket_name: str,
    bucket: dict[str, Any],
    seeds: list[int],
    out_path: str | Path,
    errors_path: str | Path,
    max_per_task: int | None,
) -> None:
    seen_state_hashes: set[str] = set()
    task_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()

    with BrowserGymEnv(base_url=browsergym_url) as env:
        for task_name in bucket["tasks"]:
            emitted_for_task = 0
            for seed in seeds:
                if max_per_task is not None and emitted_for_task >= max_per_task:
                    break

                try:
                    result = env.reset(seed=seed, task_name=task_name)
                    episode_id = f"{task_name}_seed_{seed}_{uuid4().hex[:8]}"
                    record = RawStateRecord.from_observation(
                        task_name=task_name,
                        task_bucket=bucket_name,
                        seed=seed,
                        episode_id=episode_id,
                        step_index=0,
                        observation=result.observation,
                    )

                    if record.state_hash in seen_state_hashes:
                        duplicate_counts[task_name] += 1
                        continue

                    seen_state_hashes.add(record.state_hash)
                    append_jsonl(out_path, record.to_dict())
                    task_counts[task_name] += 1
                    emitted_for_task += 1
                except Exception as exc:  # noqa: BLE001 - collection should continue
                    error_counts[task_name] += 1
                    append_jsonl(
                        errors_path,
                        {
                            "record_type": "collection_error",
                            "task_name": task_name,
                            "task_bucket": bucket_name,
                            "seed": seed,
                            "error": str(exc),
                        },
                    )

    print("Collection complete")
    print(f"Output: {out_path}")
    print(f"Errors: {errors_path}")
    print(f"Unique states: {sum(task_counts.values())}")
    print(f"Duplicates skipped: {sum(duplicate_counts.values())}")
    print(f"Errors: {sum(error_counts.values())}")
    for task_name in bucket["tasks"]:
        print(
            f"{task_name}: kept={task_counts[task_name]} "
            f"duplicates={duplicate_counts[task_name]} errors={error_counts[task_name]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect unannotated MiniWoB raw states for SFT teacher labeling."
    )
    parser.add_argument(
        "--browsergym-url",
        default=os.environ.get("BROWSERGYM_URL", "https://krish-ckpt-browsergym-v2.hf.space"),
        help="BrowserGym OpenEnv server URL.",
    )
    parser.add_argument(
        "--task-buckets",
        default="data/task_registry/task_buckets.yaml",
        help="YAML file containing task bucket definitions.",
    )
    parser.add_argument("--bucket", required=True, help="Task bucket to collect.")
    parser.add_argument(
        "--seeds",
        default="0:100",
        help="Seed range like 0:100, or comma-separated seeds like 1,2,3.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSONL path. Defaults to data/raw_states/{bucket}.jsonl.",
    )
    parser.add_argument(
        "--errors-out",
        default=None,
        help="Collection error JSONL path. Defaults to data/raw_states/{bucket}_errors.jsonl.",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=None,
        help="Optional cap on kept unique states per task.",
    )
    args = parser.parse_args()

    bucket = load_bucket(args.task_buckets, args.bucket)
    seeds = parse_seed_range(args.seeds)
    out_path = args.out or f"data/raw_states/{args.bucket}.jsonl"
    errors_path = args.errors_out or f"data/raw_states/{args.bucket}_errors.jsonl"

    collect_states(
        browsergym_url=args.browsergym_url,
        bucket_name=args.bucket,
        bucket=bucket,
        seeds=seeds,
        out_path=out_path,
        errors_path=errors_path,
        max_per_task=args.max_per_task,
    )


if __name__ == "__main__":
    main()
