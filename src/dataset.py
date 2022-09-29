import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BuildDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.image_path = df["image_path"].tolist()
        self.mask_path = df["mask_path"].tolist()
        self.transfroms = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.image_path[idx]
        mask_path = self.mask_path[idx]
        img = []
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32")
        img /= 255.0

        mask[:, :, 0][mask[:, :, 0] == 6] = 1
        mask[:, :, 0][mask[:, :, 0] == 7] = 0
        mask[:, :, 0][mask[:, :, 0] == 10] = 0
        mask[:, :, 1][mask[:, :, 1] == 7] = 1
        mask[:, :, 1][mask[:, :, 1] == 6] = 0
        mask[:, :, 1][mask[:, :, 1] == 10] = 0
        mask[:, :, 2][mask[:, :, 2] == 10] = 1
        mask[:, :, 2][mask[:, :, 2] == 7] = 0
        mask[:, :, 2][mask[:, :, 2] == 6] = 0

        transformed = self.transfroms(image=img, mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        transformed_image = np.transpose(transformed_image, (2, 1, 0))
        transformed_mask = np.transpose(transformed_mask, (2, 1, 0))

        return torch.tensor(transformed_image), torch.tensor(transformed_mask)
