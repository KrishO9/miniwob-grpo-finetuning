from dataclasses import dataclass, field
from typing import Any

from openenv import GenericEnvClient


@dataclass
class BrowserGymAction:
    action_str: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrowserGymObservation:
    text: str = ""
    url: str = ""
    screenshot: Any = None
    goal: str = ""
    axtree_txt: str = ""
    pruned_html: str = ""
    error: str = ""
    last_action_error: bool = False
    done: bool = False
    reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _observation_from_payload(payload: dict[str, Any]) -> BrowserGymObservation:
    return BrowserGymObservation(
        text=payload.get("text", ""),
        url=payload.get("url", ""),
        screenshot=payload.get("screenshot"),
        goal=payload.get("goal", ""),
        axtree_txt=payload.get("axtree_txt", ""),
        pruned_html=payload.get("pruned_html", ""),
        error=payload.get("error", ""),
        last_action_error=bool(payload.get("last_action_error", False)),
        done=bool(payload.get("done", False)),
        reward=payload.get("reward"),
        metadata=payload.get("metadata", {}),
    )


class BrowserGymEnv:
    """Synchronous BrowserGym client backed by OpenEnv's generic WebSocket client."""

    def __init__(self, base_url: str):
        self._client = GenericEnvClient(base_url=base_url).sync()
        self._client.connect()

    def reset(self, **kwargs: Any):
        result = self._client.reset(**kwargs)
        result.observation = _observation_from_payload(result.observation)
        return result

    def step(self, action: BrowserGymAction):
        result = self._client.step(
            {
                "action_str": action.action_str,
                "metadata": action.metadata,
            }
        )
        result.observation = _observation_from_payload(result.observation)
        return result

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
