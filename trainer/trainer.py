import os
import torch
from torch.utils.data import DataLoader


class Trainer:
    """
    Trainer 类：负责训练逻辑（step级别、epoch内逻辑、策略）
    支持 3D 自编码器和 2.5D 切片插值两种模式
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        save_interval: int = 10,
        model_name: str = "model",
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        self.model_name = model_name

    def train_step(self, batch) -> float:
        """单步训练，支持 3D (tensor) 和 2.5D (dict) 两种格式"""
        if self.model_name == "AE_2p5D":
            # 2.5D: {"input": [B,2,H,W], "target": [B,1,H,W]}
            x = batch["input"].to(self.device)
            target = batch["target"].to(self.device)
        elif self.model_name == "AE_3D":
            # 3D: [B,1,D,H,W] 自编码器
            x = batch.to(self.device)
            target = x
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        recon = self.model(x)
        loss = self.criterion(recon, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, dataloader: DataLoader, epoch: int, total_epochs: int) -> float:
        """单个 epoch 的训练"""
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            loss = self.train_step(batch)
            total_loss += loss

            print(
                f"Epoch [{epoch+1}/{total_epochs}] "
                f"Batch [{batch_idx+1}/{len(dataloader)}] "
                f"Loss: {loss:.6f}"
            )

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{total_epochs}] Avg Loss: {avg_loss:.6f}")

        return avg_loss

    def save_checkpoint(self, epoch: int, loss: float):
        """保存检查点"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_epoch{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")

    def fit(self, dataloader: DataLoader, epochs: int):
        """完整训练流程"""
        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader, epoch, epochs)

            # 按间隔保存检查点
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch + 1, avg_loss)
