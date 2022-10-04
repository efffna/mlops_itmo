from pathlib import Path

import pandas as pd
from config import AppConfig


def main_actions(config: AppConfig):
    dataset_path = config.dataset_path
    root = Path(f"{dataset_path}")

    mask_path = sorted(list(root.glob("mask/*")))
    image_path = sorted(list(root.glob("image/*")))

    df = pd.DataFrame()
    df["image_path"] = image_path
    df["mask_path"] = mask_path
    df.to_csv(f"{dataset_path}/data.csv", index=False)


def main():
    config = AppConfig.parse_raw()
    main_actions(config=config)


if __name__ == "__main__":
    main()
