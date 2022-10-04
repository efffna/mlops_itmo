import pandas as pd
import torch.optim as optim
from config import AppConfig

from src.train import Train
from src.utils import build_model, fetch_scheduler


def train_model(config: AppConfig):
    trainer = Train(config)
    # config
    backbone = config.backbone
    num_classes = config.num_classes
    in_channels = config.in_channels
    device = config.device
    lr = config.lr
    wd = config.weight_decay
    scheduler = config.scheduler
    epochs = config.epochs
    batch_size = config.batch_size

    train_loader, valid_loader = trainer.prepare_loaders()
    model = build_model(backbone, num_classes, in_channels, device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = fetch_scheduler(optimizer, scheduler, batch_size=batch_size, epochs=epochs)

    model, history = trainer.run_training(
        model,
        optimizer,
        scheduler,
        device=device,
        num_epochs=epochs,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )


if __name__ == "__main__":
    config = AppConfig.parse_raw()
    train_model(config)
