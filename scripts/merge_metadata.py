# -*- coding: utf-8 -*-
"""
合并元数据与 AE3D 特征。

用法:
    python scripts/merge_metadata.py --metadata data/test.csv \
        --features temp/ae3d_epoch50_test_features.csv \
        --output temp/test_merged.csv
"""

import os
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="合并元数据与 AE3D 特征")
    parser.add_argument("--metadata", type=str, required=True, help="原始数据 CSV")
    parser.add_argument("--features", type=str, required=True, help="AE3D 特征 CSV")
    parser.add_argument("--output", type=str, required=True, help="输出 CSV")
    parser.add_argument("--drop_old", type=int, default=1, help="删除旧特征数量 (0=不删)")
    parser.add_argument("--old_prefix", type=str, default="feature_", help="旧特征前缀")
    args = parser.parse_args()

    # 读取
    meta_df = pd.read_csv(args.metadata)
    feat_df = pd.read_csv(args.features)
    print(f"元数据: {meta_df.shape}, 特征: {feat_df.shape}")

    # 删除旧特征列（如果指定）
    if args.drop_old > 0:
        old_cols = [f"{args.old_prefix}{i}" for i in range(args.drop_old)]
        meta_df = meta_df.drop(columns=[c for c in old_cols if c in meta_df.columns])
        print(f"删除旧特征后: {meta_df.shape}")

    # 合并
    merged = meta_df.merge(feat_df, on="patient_id", how="inner")
    print(f"合并结果: {merged.shape}")

    # 保存
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"已保存: {args.output}")


if __name__ == "__main__":
    main()
