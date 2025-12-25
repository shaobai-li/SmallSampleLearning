import torch
import torch.nn as nn
import torch.nn.functional as F

class L1ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, recon: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return F.l1_loss(recon, target, reduction="mean")

def l1_reconstruction_loss(
    recon: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return L1ReconstructionLoss()(recon, target)