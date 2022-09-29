import copy
import gc
import time
from collections import defaultdict

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import BuildDataset
from src.utils import get_augmentations, criterion, dice_coef, iou_coef

c_ = Fore.GREEN
sr_ = Style.RESET_ALL


class Trainer:
    def __init__(self, config):
        self.dataset_path = config.dataset_path
        self.dir_csv = f"{self.dataset_path}/data.csv"
        self.data_df = pd.read_csv(self.dir_csv)
        self.save_models = config.save_models
        self.batch_size = config.batch_size
        self.size_image = config.size_image
        self.workers = config.workers
        self.device = torch.device(config.device)
        self.n_accumulate = max(1, 32 // self.batch_size)

    def prepare_loaders(self):
        data_transforms = get_augmentations(self.size_image)
        train_dataset, valid_dataset = train_test_split(
            self.data_df, test_size=0.1, random_state=43
        )
        train_dataset = BuildDataset(train_dataset, transform=data_transforms["train"])
        valid_dataset = BuildDataset(valid_dataset, transform=data_transforms["test"])
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=False,
            pin_memory=True,
        )

        return train_loader, valid_loader

    def train_one_epoch(self, model, optimizer, scheduler, device, epoch, train_loader):
        model.train()
        scaler = amp.GradScaler()
        running_loss = 0.0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train")
        for step, (images, masks) in pbar:
            images = images.to(device, dtype=torch.float)

            masks = masks.to(device, dtype=torch.float)
            batch_size = images.size(0)

            with amp.autocast(enabled=True):
                y_pred = model(images)
                loss = criterion(y_pred, masks)
                loss = loss / self.n_accumulate

            scaler.scale(loss).backward()

            if (step + 1) % self.n_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            running_loss += loss.item() * batch_size
            epoch_loss = running_loss / batch_size
            mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                train_loss=f"{epoch_loss:0.4f}", lr=f"{current_lr:0.5f}", gpu_mem=f"{mem:0.2f} GB"
            )
        torch.cuda.empty_cache()
        gc.collect()
        return epoch_loss

    def valid_one_epoch(self, model, device, epoch, valid_loader):
        model.eval()
        running_loss = 0.0
        val_scores = []
        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Valid ")
        for step, (images, masks) in pbar:
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)
            batch_size = images.size(0)

            y_pred = model(images)
            loss = criterion(y_pred, masks)
            running_loss += loss.item() * batch_size
            epoch_loss = running_loss / batch_size

            y_pred = nn.Sigmoid()(y_pred)
            val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
            val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
            val_scores.append([val_dice, val_jaccard])
            mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            pbar.set_postfix(
                valid_loss=f"{epoch_loss:0.4f}",
                gpu_memory=f"{mem:0.2f} GB",
            )
        val_scores = np.mean(val_scores, axis=0)
        torch.cuda.empty_cache()
        gc.collect()

        return epoch_loss, val_scores

    def run_training(
        self, model, optimizer, scheduler, device, num_epochs, train_loader, valid_loader
    ):
        start = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_dice = -np.inf
        history = defaultdict(list)

        for epoch in range(1, num_epochs + 1):
            gc.collect()
            print(f"Epoch {epoch}/{num_epochs}", end="")
            train_loss = self.train_one_epoch(
                model,
                optimizer,
                scheduler,
                device=self.device,
                epoch=epoch,
                train_loader=train_loader,
            )

            val_loss, val_scores = self.valid_one_epoch(
                model, device=self.device, epoch=epoch, valid_loader=valid_loader
            )
            val_dice, val_jaccard = val_scores
            history["Train Loss"].append(train_loss)
            history["Valid Loss"].append(val_loss)
            history["Valid Dice"].append(val_dice)
            history["Valid Jaccard"].append(val_jaccard)
            print()
            print(f"mIoU: {val_jaccard:0.4f} Dice: {val_dice:0.4f}  ")
            if val_dice >= best_dice:
                print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
                best_dice = val_dice
                best_jaccard = val_jaccard
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f"{self.save_models}/best.pt")

            last_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f"{self.save_models}/last.pt")

        end = time.time()
        time_elapsed = end - start
        print(
            "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
                time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60
            )
        )
        print("Best Score: {:.4f}".format(best_jaccard))
        model.load_state_dict(best_model_wts)

        return model, history
