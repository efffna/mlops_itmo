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
import tritonclient.grpc as grpcclient


class InferenceModel:
    def __init__(self, url):
        self.client = grpcclient.InferenceServerClient(url=url, verbose=True)
        self.model_onnx_path = "models/effb2.onnx"
        self.test_path = sorted(list(("data/test/*")))
        self.size_image = 224
        self.df = pd.DataFrame()
        self.df["test_path"] = self.test_pat
        self._load_model(self.model_onnx_path)

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
                pred = self.triton_process_input(imgs)
                pred = torch.tensor(pred)
                pred = (nn.Sigmoid()(pred) > 0.5).double()
            preds.append(pred)
            preds = torch.mean(torch.stack(preds, dim=0), dim=0).cpu().detach()
            preds = np.squeeze(preds)
            self.save_result(preds, i)

    def triton_process_input(self, image: np.ndarray):
        inputs = []
        inputs.append(grpcclient.InferInput(self.model_inputs, image.shape, "FP32"))
        inputs[0].set_data_from_numpy(image)
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput(self.model_outputs))

        return self.client.infer(self.model_name, inputs, outputs=outputs).as_numpy(
            self.model_outputs
        )


if __name__ == "__main__":
    inference = InferenceModel(url="0.0.0.0:8001")
    inference.predict()
