import argparse
import torch
from torch.utils.data import DataLoader

from datasets.dataset import NRRDDataset
from models.model import AE_3D
from losses.loss import L1ReconstructionLoss
from trainer.trainer import Trainer


def train(args):
    """组装一切 + 启动训练"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & DataLoader
    dataset = NRRDDataset(args.csv_path, args.nrrd_dir, bin_factor=args.bin_factor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Loaded {len(dataset)} samples")

    # Model, Loss, Optimizer
    model = AE_3D().to(device)
    criterion = L1ReconstructionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval,
    )

    # 启动训练
    trainer.fit(dataloader, args.epochs)


def main():
    parser = argparse.ArgumentParser(description="Train AE_3D")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to train_ssl.csv")
    parser.add_argument("--nrrd_dir", type=str, required=True, help="Directory with NRRD files")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bin_factor", type=int, default=2, help="Binning factor for downsampling")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
