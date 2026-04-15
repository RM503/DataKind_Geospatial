import tomllib
from pathlib import Path
from typing import Optional

def load_config(config_path: Path | str) -> Optional[dict]:
    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        return config
    except FileNotFoundError:
        return None