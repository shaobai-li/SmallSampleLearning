#!/usr/bin/env python3
"""
检查 NRRD 数据是否存在 nan/inf 或常量值问题
用于排查训练时 loss 变成 nan 的原因
"""
import nrrd
import numpy as np
import pandas as pd
import os
import argparse


def check_nrrd_data(csv_path: str, nrrd_dir: str):
    """
    检查所有 NRRD 文件的数据质量
    
    Args:
        csv_path: CSV 文件路径，包含 patient_id 列
        nrrd_dir: NRRD 文件所在目录
    """
    df = pd.read_csv(csv_path)
    
    total = 0
    missing = 0
    problems = 0
    
    print(f"Checking {len(df)} patients from {csv_path}")
    print(f"NRRD directory: {nrrd_dir}")
    print("-" * 60)
    
    for pid in df["patient_id"]:
        path = os.path.join(nrrd_dir, f"{pid}.nrrd")
        total += 1
        
        if not os.path.exists(path):
            print(f"[MISSING] {pid}: file not found")
            missing += 1
            continue
        
        try:
            data, _ = nrrd.read(path)
            
            has_nan = np.any(np.isnan(data))
            has_inf = np.any(np.isinf(data))
            is_constant = data.max() == data.min()
            
            if has_nan or has_inf:
                nan_count = np.isnan(data).sum()
                inf_count = np.isinf(data).sum()
                print(f"[PROBLEM] {pid}: nan={nan_count}, inf={inf_count}")
                problems += 1
            elif is_constant:
                print(f"[PROBLEM] {pid}: constant value {data.max()}")
                problems += 1
            else:
                # 可选：打印数据范围
                # print(f"[OK] {pid}: range=[{data.min():.2f}, {data.max():.2f}], shape={data.shape}")
                pass
                
        except Exception as e:
            print(f"[ERROR] {pid}: {e}")
            problems += 1
    
    print("-" * 60)
    print(f"Summary: total={total}, missing={missing}, problems={problems}")
    print(f"OK: {total - missing - problems}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check NRRD data for nan/inf/constant values")
    parser.add_argument("--csv", type=str, default="data/train_ssl.csv", help="CSV file path")
    parser.add_argument("--nrrd_dir", type=str, default="/data/res/res3d/inbox/02_masked/", help="NRRD directory")
    
    args = parser.parse_args()
    check_nrrd_data(args.csv, args.nrrd_dir)

