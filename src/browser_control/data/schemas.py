from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import hashlib
import json
import re
from typing import Any


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().lower()


def stable_hash(*parts: str) -> str:
    payload = "\n".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_dumps(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=True, sort_keys=True)


@dataclass
class RawStateRecord:
    record_type: str
    task_name: str
    task_bucket: str
    seed: int
    episode_id: str
    step_index: int
    goal: str
    axtree_txt: str
    url: str
    state_hash: str
    goal_hash: str
    axtree_hash: str
    done: bool
    reward: float | None
    error: str
    last_action_error: bool
    metadata: dict[str, Any]

    @classmethod
    def from_observation(
        cls,
        *,
        task_name: str,
        task_bucket: str,
        seed: int,
        episode_id: str,
        step_index: int,
        observation: Any,
    ) -> RawStateRecord:
        goal = getattr(observation, "goal", "") or ""
        axtree_txt = getattr(observation, "axtree_txt", "") or ""
        url = getattr(observation, "url", "") or ""
        metadata = getattr(observation, "metadata", {}) or {}
        browsergym_info = metadata.get("browsergym_info", {})

        goal_hash = stable_hash(normalize_text(goal))
        axtree_hash = stable_hash(normalize_text(axtree_txt))
        state_hash = stable_hash(task_name, goal_hash, axtree_hash)

        return cls(
            record_type="raw_state",
            task_name=task_name,
            task_bucket=task_bucket,
            seed=seed,
            episode_id=episode_id,
            step_index=step_index,
            goal=goal,
            axtree_txt=axtree_txt,
            url=url,
            state_hash=state_hash,
            goal_hash=goal_hash,
            axtree_hash=axtree_hash,
            done=bool(getattr(observation, "done", False)),
            reward=getattr(observation, "reward", None),
            error=getattr(observation, "error", "") or "",
            last_action_error=bool(getattr(observation, "last_action_error", False)),
            metadata={
                "browsergym_info": browsergym_info,
                "collection_time": datetime.now(UTC).isoformat(),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
