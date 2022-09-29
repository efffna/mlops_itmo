import albumentations as A
import cv2
import segmentation_models_pytorch as smp
import torch
from torch.optim import lr_scheduler

JaccardLoss = smp.losses.JaccardLoss(mode="multilabel")
DiceLoss = smp.losses.DiceLoss(mode="multilabel")
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss = smp.losses.LovaszLoss(mode="multilabel", per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode="multilabel", log_loss=False)


def build_model(backbone, num_classes, in_channels, device):
    model = smp.FPN(
        encoder_name=backbone,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )
    model.to(device)
    return model


def augmentations(image_size):
    data_transforms = {
        "train": A.Compose(
            [
                A.Resize(*image_size, interpolation=cv2.INTER_NEAREST),
                # A.RandomBrightnessContrast(brightness_limit=(0.0,0.5), contrast_limit=(0.0, 0.3), p=0.7),
                # A.CLAHE(clip_limit=(1,10), p=1)
            ],
            p=1.0,
        ),
        "test": A.Compose(
            [
                A.Resize(*image_size),
            ],
            p=1.0,
        ),
    }

    return data_transforms


def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


def criterion(y_pred, y_true):
    return 0.5 * BCELoss(y_pred, y_true) + 0.5 * TverskyLoss(y_pred, y_true)


def fetch_scheduler(optimizer, scheduler, batch_size, epochs, min_lr=1e-6, T_0=25):
    T_max = int(30000 / batch_size * epochs) + 50

    if scheduler == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr)

    elif scheduler == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=min_lr)

    elif scheduler == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=7,
            threshold=0.0001,
            min_lr=min_lr,
        )
    elif scheduler == "ExponentialLR":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    elif scheduler == None:
        return None

    return scheduler
