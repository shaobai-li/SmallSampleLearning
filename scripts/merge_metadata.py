# -*- coding: utf-8 -*-
"""
合并元数据与 AE3D 特征。

用法:
    python scripts/merge_metadata.py --config configs/merge_metadata.yaml
"""

import os
import argparse
import pandas as pd
from omegaconf import OmegaConf


def merge_metadata(cfg):
    # 读取
    meta_df = pd.read_csv(cfg.data.metadata_path)
    
    print(f"元数据: {meta_df.shape}")
    for prefix in cfg.merge.old_prefix:
        old_cols = [c for c in meta_df.columns if c.startswith(prefix)]
        print(f"旧特征数: {len(old_cols)}")
        meta_df = meta_df.drop(columns=old_cols)
        print(f"删除旧特征后: {meta_df.shape}")

    for feat_path in cfg.data.features_path:
        feat_df = pd.read_csv(feat_path)
        print(f"特征: {feat_df.shape}")
        meta_df = meta_df.merge(feat_df, on="patient_id", how="inner")
        print(f"合并后: {meta_df.shape}")



def main():
    parser = argparse.ArgumentParser(description="合并元数据与 AE3D 特征")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    # 加载配置
    cfg = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(cfg))

    merge_metadata(cfg)


if __name__ == "__main__":
    main()
