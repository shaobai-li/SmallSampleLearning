import os
import pandas as pd
import nrrd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class Dataset3D(Dataset):
    """3D Volume Dataset for Autoencoder"""

    def __init__(self, csv_path: str, nrrd_dir: str, bin_factor: int = 1):
        df = pd.read_csv(csv_path)
        self.patient_ids = df["patient_id"].tolist()
        self.nrrd_dir = nrrd_dir
        self.bin_factor = bin_factor

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        nrrd_path = os.path.join(self.nrrd_dir, f"{patient_id}.nrrd")

        data, _ = nrrd.read(nrrd_path)
        # [D, H, W] -> [1, D, H, W]
        tensor = torch.from_numpy(data).float().unsqueeze(0)

        # Binning
        if self.bin_factor > 1:
            tensor = tensor.unsqueeze(0)  # [1, 1, D, H, W]
            tensor = F.avg_pool3d(tensor, kernel_size=self.bin_factor)
            tensor = tensor.squeeze(0)    # [1, D', H', W']

        return tensor