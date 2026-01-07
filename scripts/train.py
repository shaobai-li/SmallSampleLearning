import argparse
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets import Dataset3D, Dataset2p5D
from models import AE_3D, AE_2p5D
from losses.loss import L1ReconstructionLoss
from trainer.trainer import Trainer


def train(cfg):
    """组装一切 + 启动训练"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 根据配置选择 Dataset 和 Model
    model_name = cfg.model.name
    if model_name == "AE_2p5D":
        dataset = Dataset2p5D(
            cfg.data.csv_path,
            cfg.data.nrrd_dir,
            bin_factor=cfg.data.bin_factor,
        )
        model = AE_2p5D().to(device)
    else:
        dataset = Dataset3D(
            cfg.data.csv_path,
            cfg.data.nrrd_dir,
            bin_factor=cfg.data.bin_factor,
        )
        model = AE_3D().to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    print(f"Loaded {len(dataset)} samples, Model: {model_name}")
    criterion = L1ReconstructionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)

    # Trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=cfg.checkpoint.dir,
        save_interval=cfg.checkpoint.save_interval,
    )

    # 启动训练
    trainer.fit(dataloader, cfg.training.epochs)


def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder (3D/2.5D)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    # 加载配置
    cfg = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(cfg))

    train(cfg)


if __name__ == "__main__":
    main()
