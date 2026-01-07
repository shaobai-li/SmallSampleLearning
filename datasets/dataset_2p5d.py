import os
import pandas as pd
import nrrd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class Dataset2p5D(Dataset):
    """
    2.5D Slice Interpolation Dataset
    Input: [S(z-1), S(z+1)] → Target: S(z)
    """

    def __init__(self, csv_path: str, nrrd_dir: str, bin_factor: int = 1, min_nonzero_frac: float = 0.0):
        df = pd.read_csv(csv_path)
        self.patient_ids = df["patient_id"].tolist()
        self.nrrd_dir = nrrd_dir
        self.bin_factor = bin_factor
        self.min_nonzero_frac = float(min_nonzero_frac)
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        dropped = 0

        for pid in self.patient_ids:
            nrrd_path = os.path.join(self.nrrd_dir, f"{pid}.nrrd")
            if not os.path.exists(nrrd_path):
                continue

            data, _ = nrrd.read(nrrd_path)  # [D,H,W] 或 [H,W,D]（取决于你的数据）
            # 统一按你的 Dataset 写法：data[z] 是一个 2D 切片
            depth = data.shape[0]

            for z in range(1, depth - 1):
                inp = np.stack([data[z - 1], data[z + 1]], axis=0)
                nonzero_frac = np.mean(inp != 0)

                if nonzero_frac <= self.min_nonzero_frac:
                    dropped += 1
                    continue

                samples.append((pid, z))

        print(f"[Dataset2p5D] built samples={len(samples)}, dropped={dropped}, min_nonzero_frac={self.min_nonzero_frac}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, z = self.samples[idx]
        nrrd_path = os.path.join(self.nrrd_dir, f"{pid}.nrrd")

        data, _ = nrrd.read(nrrd_path)  # [D, H, W]

        # 提取三切片
        slice_prev = data[z - 1]
        slice_curr = data[z]
        slice_next = data[z + 1]

        # Input: [2, H, W], Target: [1, H, W]
        input_slices = torch.from_numpy(
            np.stack([slice_prev, slice_next], axis=0)
        ).float()
        target_slice = torch.from_numpy(slice_curr).float().unsqueeze(0)

        # Binning
        if self.bin_factor > 1:
            input_slices = F.avg_pool2d(
                input_slices.unsqueeze(0), kernel_size=self.bin_factor
            ).squeeze(0)
            target_slice = F.avg_pool2d(
                target_slice.unsqueeze(0), kernel_size=self.bin_factor
            ).squeeze(0)

        return {
            "input": input_slices,
            "target": target_slice,
        }