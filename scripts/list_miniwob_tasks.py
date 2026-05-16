from __future__ import annotations

import argparse
from pathlib import Path


def discover_miniwob_tasks() -> list[str]:
    import browsergym.miniwob  # noqa: F401 - registers MiniWoB environments
    import gymnasium as gym

    env_ids = []
    for env_id in gym.envs.registry.keys():
        env_id_str = str(env_id)
        if "miniwob" in env_id_str.lower():
            env_ids.append(env_id_str)
    return sorted(set(env_ids))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List MiniWoB task ids registered in the local BrowserGym install."
    )
    parser.add_argument(
        "--out",
        default="data/task_registry/miniwob_tasks.txt",
        help="Output text file for discovered task ids.",
    )
    args = parser.parse_args()

    try:
        tasks = discover_miniwob_tasks()
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import BrowserGym MiniWoB locally. Install browsergym-miniwob "
            "in this environment or run the collector against known task names."
        ) from exc

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(tasks) + "\n", encoding="utf-8")

    print(f"Discovered {len(tasks)} MiniWoB task ids")
    print(f"Wrote {output_path}")
    for task in tasks:
        print(task)


if __name__ == "__main__":
    main()
