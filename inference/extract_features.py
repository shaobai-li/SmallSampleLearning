import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm

from train_ssl.data.dataset import NRRDDataset
from train_ssl.models.model import AE_3D


def extract_features(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = AE_3D().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load dataset
    dataset = NRRDDataset(args.csv_path, args.nrrd_dir, bin_factor=args.bin_factor)
    print(f"Loaded {len(dataset)} samples")

    # Extract features
    patient_ids = dataset.patient_ids
    features = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Extracting"):
            idx = int(idx)
            x = dataset[idx].unsqueeze(0).to(device)  # [1, 1, D, H, W]
            feat = model.get_latent(x)  # [1, 256]
            features.append(feat.cpu().numpy().flatten())

    # Build DataFrame
    feat_cols = [f"feat_{i}" for i in range(256)]
    df = pd.DataFrame(features, columns=feat_cols)
    df.insert(0, "patient_id", patient_ids)

    # Save
    df.to_csv(args.output, index=False)
    print(f"Saved features to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Extract features using trained AE_3D")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to train_ssl.csv")
    parser.add_argument("--nrrd_dir", type=str, required=True, help="Directory with NRRD files")
    parser.add_argument("--bin_factor", type=int, default=2, help="Binning factor (should match training)")
    parser.add_argument("--output", type=str, default="features.csv", help="Output CSV path")

    args = parser.parse_args()
    extract_features(args)


if __name__ == "__main__":
    main()

