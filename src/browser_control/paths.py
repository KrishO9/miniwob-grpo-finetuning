import os
from pathlib import Path


def get_path_to_configs() -> str:
    path = str(Path(__file__).parent.parent.parent / "configs")

    # create path if it does not exist
    Path(path).mkdir(parents=True, exist_ok=True)

    return path

def get_path_to_media() -> str:
    path = str(Path(__file__).parent.parent.parent / "media")

    # create path if it does not exist
    Path(path).mkdir(parents=True, exist_ok=True)

    return path

def get_path_model_checkpoints(
    experiment_name: str,
) -> str:
    """
    Returns path to model checkpoints.

    Defaults to the Modal volume path used by the original example. Set
    MODEL_CHECKPOINT_ROOT=/kaggle/working/model_checkpoints for Kaggle.
    """
    checkpoint_root = os.environ.get("MODEL_CHECKPOINT_ROOT", "/model_checkpoints")
    path = Path(checkpoint_root) / experiment_name.replace("/", "--")

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    return str(path)
