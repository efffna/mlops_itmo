from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnx
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import tqdm
from onnxruntime import InferenceSession

from config import AppConfig


class InferenceModel:
    def __init__(self, config):
        self.model_onnx_path = config.model_onnx_path
        self._load_model(self.model_onnx_path)
        self.size_image = config.size_image
        root = Path(f"{config.dataset_path}")
        self.test_path = sorted(list(root.glob("test/*")))
        self.df = pd.DataFrame()
        self.df["test_path"] = self.test_path

    def _load_model(self, model_path: Path):
        model = onnx.load(model_path)
        self.session = InferenceSession(model.SerializeToString())

    def _preprocess_data(self, path):
        input_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype("float32")
        img /= 255.0
        img = np.transpose(img, (2, 1, 0))
        img = np.expand_dims(img, 0)
        img = torch.tensor(img)
        img = img.to(torch.device("cpu"), dtype=torch.float)
        return img

    def save_result(self, preds, count):
        plt.imshow(preds.permute((2, 1, 0)))
        plt.savefig(f"output/{count}.png", bbox_inches="tight")

    def predict(self):
        for i in tqdm.tqdm(range(0, len(self.df))):
            path = self.test_path[i].as_posix()
            img = self._preprocess_data(path)
            preds = []
            with torch.no_grad():
                imgs = img.cpu().numpy()
                pred = self.session.run(None, {self.session.get_inputs()[0].name: imgs})
                pred = torch.tensor(pred)
                pred = (nn.Sigmoid()(pred) > 0.5).double()
            preds.append(pred)
            preds = torch.mean(torch.stack(preds, dim=0), dim=0).cpu().detach()
            preds = np.squeeze(preds)
            self.save_result(preds, i)


if __name__ == "__main__":
    config = AppConfig.parse_raw()
    inference = InferenceModel(config)
    inference.predict()