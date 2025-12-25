import os
import pandas as pd
import nrrd
import torch
from torch.utils.data import Dataset


class NRRDDataset(Dataset):
    """
    Dataset for loading NRRD files for AE_3D training.
    """

    def __init__(self, csv_path: str, nrrd_dir: str):
        """
        Args:
            csv_path: Path to CSV file with patient_id column.
            nrrd_dir: Directory containing {patient_id}.nrrd files.
        """
        df = pd.read_csv(csv_path)
        self.patient_ids = df["patient_id"].tolist()
        self.nrrd_dir = nrrd_dir

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        nrrd_path = os.path.join(self.nrrd_dir, f"{patient_id}.nrrd")

        data, _ = nrrd.read(nrrd_path)
        # [D, H, W] -> [1, D, H, W]
        tensor = torch.from_numpy(data).float().unsqueeze(0)

        return tensor

