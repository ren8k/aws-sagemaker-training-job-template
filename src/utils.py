import yaml
import pandas as pd
from pathlib import Path


def load_config(config_path):
    with open(config_path, "r") as stream:
        return yaml.safe_load(stream)


def log(data: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        df: pd.DataFrame = pd.read_csv(path)
        df = pd.DataFrame(df.to_dict("records") + [data])
        df.to_csv(path, index=False)
    else:
        pd.DataFrame([data]).to_csv(path, index=False)
