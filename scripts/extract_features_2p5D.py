import os
import argparse

import numpy as np
import nrrd
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from models import AE_2p5D


def _aggregate_slice_features(x: torch.Tensor, method: str) -> torch.Tensor:
    """
    Args:
        x: [N, C]
        method: "mean" | "max"
    Returns:
        [C]
    """
    method = (method or "mean").lower()
    if method == "mean":
        return x.mean(dim=0)
    if method == "max":
        return x.max(dim=0).values
    raise ValueError(f"Unsupported aggregate method: {method}")


def extract_features_2p5d(cfg):
    """加载 AE_2p5D 并按病人从 nrrd 提取 patient-level 特征"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------- config --------
    nrrd_dir = cfg.data.nrrd_dir
    bin_factor = int(cfg.data.get("bin_factor", 1))
    min_nonzero_frac = float(cfg.data.get("min_nonzero_frac", 0.0))

    extract_cfg = cfg.get("extract", {})
    batch_size = int(extract_cfg.get("batch_size", 64))
    aggregate = str(extract_cfg.get("aggregate", "mean"))

    # -------- model --------
    model = AE_2p5D().to(device)
    checkpoint = torch.load(cfg.model.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {cfg.model.checkpoint}")

    # -------- patient list --------
    df_ids = pd.read_csv(cfg.data.csv_path)
    patient_ids = df_ids["patient_id"].tolist()

    features = []
    with torch.no_grad():
        for pid in tqdm(patient_ids, desc="Extracting 2.5D"):
            nrrd_path = os.path.join(nrrd_dir, f"{pid}.nrrd")

            if not os.path.exists(nrrd_path):
                features.append(np.zeros((256,), dtype=np.float32))
                continue

            data, _ = nrrd.read(nrrd_path)  # 约定: data[z] 是 2D slice
            depth = int(data.shape[0])

            windows = []
            for z in range(1, depth - 1):
                inp = np.stack([data[z - 1], data[z + 1]], axis=0)  # [2,H,W]
                nonzero_frac = float(np.mean(inp != 0))
                if nonzero_frac <= min_nonzero_frac:
                    continue
                windows.append(inp)

            if len(windows) == 0:
                features.append(np.zeros((256,), dtype=np.float32))
                continue

            slice_feats = []
            for start in range(0, len(windows), batch_size):
                chunk = np.stack(windows[start:start + batch_size], axis=0)  # [B,2,H,W]
                x = torch.from_numpy(chunk).float().to(device)
                if bin_factor > 1:
                    x = F.avg_pool2d(x, kernel_size=bin_factor)

                feat = model.get_latent(x)  # [B,256]
                slice_feats.append(feat.cpu())

            slice_feats = torch.cat(slice_feats, dim=0)  # [N,256]
            patient_feat = _aggregate_slice_features(slice_feats, aggregate).numpy().astype(np.float32)
            features.append(patient_feat)

    # -------- save --------
    feat_cols = [f"f{i}" for i in range(256)]
    out_df = pd.DataFrame(features, columns=feat_cols)
    out_df.insert(0, "patient_id", patient_ids)

    output_dir = os.path.dirname(cfg.output.path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    out_df.to_csv(cfg.output.path, index=False)
    print(f"Saved features to: {cfg.output.path}")


def main():
    parser = argparse.ArgumentParser(description="Extract features using trained AE_2p5D")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(cfg))
    extract_features_2p5d(cfg)


if __name__ == "__main__":
    main()


