from pathlib import Path
from typing import List, Union

from pydantic import HttpUrl
from pydantic_yaml import YamlModel


class AppConfig(YamlModel):
    # path
    dataset_name_default: str
    dataset_path: Path
    save_models: Path
    model_onnx_path: Path
    task_name_train: str
    task_name_preparation: str
    task_name: str
    project: str
    version: str
    dataset_id: str
    output_dataset_path: Path

    # train model
    batch_size: int
    epochs: int
    lr: float
    size_image: list
    workers: int
    device: str
    backbone: str
    num_classes: int
    in_channels: int
    weight_decay: float
    scheduler: str

    @classmethod
    def parse_raw(cls, filename: Union[str, Path] = "config.yaml", *args, **kwargs):
        with open(filename) as f:
            data = f.read()
        return super().parse_raw(data, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
