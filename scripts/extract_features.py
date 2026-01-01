import os
import argparse
import torch
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from datasets.dataset import NRRDDataset
from models.model import AE_3D


def extract_features(cfg):
    """加载模型并提取特征"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = AE_3D().to(device)
    checkpoint = torch.load(cfg.model.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {cfg.model.checkpoint}")

    # Load dataset
    dataset = NRRDDataset(
        cfg.data.csv_path,
        cfg.data.nrrd_dir,
        bin_factor=cfg.data.bin_factor,
    )
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
    feat_cols = [f"feature_{i}" for i in range(256)]
    df = pd.DataFrame(features, columns=feat_cols)
    df.insert(0, "patient_id", patient_ids)

    # Save (自动创建目录)
    output_dir = os.path.dirname(cfg.output.path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(cfg.output.path, index=False)
    print(f"Saved features to: {cfg.output.path}")


def main():
    parser = argparse.ArgumentParser(description="Extract features using trained AE_3D")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    # 加载配置
    cfg = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(cfg))

    extract_features(cfg)


if __name__ == "__main__":
    main()