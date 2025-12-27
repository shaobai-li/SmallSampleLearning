import os
import argparse
import torch
from torch.utils.data import DataLoader

from datasets.dataset import NRRDDataset
from models.model import AE_3D
from losses.loss import L1ReconstructionLoss


def train(args):
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

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, x in enumerate(dataloader):
            x = x.to(device)

            recon = model(x)
            loss = criterion(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Avg Loss: {avg_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(args.checkpoint_dir, f"ae3d_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


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

