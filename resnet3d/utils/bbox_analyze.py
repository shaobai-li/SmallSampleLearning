#!/usr/bin/env python3
import nrrd
import numpy as np
import pandas as pd
import os
import sys
import argparse
import glob


def analyze_bbox_from_nrrd(input_dir, output_csv):
    """
    功能一：分析 input 目录下所有 nrrd 文件的 bbox 距离
    """
    records = []
    
    # 获取 input 目录下所有 nrrd 文件
    nrrd_files = glob.glob(os.path.join(input_dir, "*.nrrd"))
    
    if not nrrd_files:
        print(f"Error: No NRRD files found in {input_dir}")
        sys.exit(1)
    
    # 按文件名排序
    nrrd_files.sort()
    
    for mask_path in nrrd_files:
        filename = os.path.basename(mask_path)
        print(f"# processing {filename}")
        
        try:
            mask, _ = nrrd.read(mask_path)
            
            # 直接获取 mask 内的坐标索引
            x_idx, y_idx, z_idx = np.where(mask > 0)
            
            if len(x_idx) == 0:
                print(f"Warning: {filename} has no non-zero values")
                continue
            
            x_min, x_max = np.min(x_idx), np.max(x_idx)
            y_min, y_max = np.min(y_idx), np.max(y_idx)
            z_min, z_max = np.min(z_idx), np.max(z_idx)
            
            # 计算最大距离
            d_x = x_max - x_min
            d_y = y_max - y_min
            d_z = z_max - z_min
            
            records.append({
                "filename": filename,
                "d_x": d_x, 
                "d_y": d_y, 
                "d_z": d_z,
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    if not records:
        print("Error: No valid records generated")
        sys.exit(1)
    
    df = pd.DataFrame(records)
    print(f"\nProcessed {len(df)} files")
    print(df.head())
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")


def calculate_percentiles(csv_file):
    """
    功能二：计算 csv 文件中三个方向 bbox 的 95/99 百分位数
    """
    if not os.path.exists(csv_file):
        print(f"Error: CSV file does not exist: {csv_file}")
        sys.exit(1)
    
    df = pd.read_csv(csv_file)
    
    # 检查必要的列是否存在
    required_cols = ['d_x', 'd_y', 'd_z']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: CSV file missing required columns: {missing_cols}")
        sys.exit(1)
    
    print(f"\nStatistics for {len(df)} records:")
    print("=" * 60)
    
    for col in required_cols:
        values = df[col].dropna()
        p95 = np.percentile(values, 95)
        p99 = np.percentile(values, 99)
        mean_val = values.mean()
        median_val = values.median()
        min_val = values.min()
        max_val = values.max()
        
        print(f"\n{col}:")
        print(f"  Min:     {min_val:.2f}")
        print(f"  Mean:    {mean_val:.2f}")
        print(f"  Median:  {median_val:.2f}")
        print(f"  95th percentile: {p95:.2f}")
        print(f"  99th percentile: {p99:.2f}")
        print(f"  Max:     {max_val:.2f}")


def analyze_crop_outside_voxels(input_dir, bbox_size, output_csv=None):
    """
    功能三：统计给定 bbox 下，3D crop 之外的体素个数和百分比
    以 mask 重心为中心进行 crop，统计 mask=1 但在 crop 范围外的体素
    """
    records = []
    
    # 解析 bbox_size，格式可能是 "x0,y0,z0" 或三个独立参数
    if isinstance(bbox_size, str):
        try:
            bbox_x, bbox_y, bbox_z = map(int, bbox_size.split(','))
        except ValueError:
            print(f"Error: Invalid bbox_size format. Expected format: 'x0,y0,z0' (e.g., '64,64,64')")
            sys.exit(1)
    else:
        bbox_x, bbox_y, bbox_z = bbox_size
    
    print(f"Bounding box size: ({bbox_x}, {bbox_y}, {bbox_z})")
    print(f"Processing files in: {input_dir}\n")
    
    # 获取 input 目录下所有 nrrd 文件
    nrrd_files = glob.glob(os.path.join(input_dir, "*.nrrd"))
    
    if not nrrd_files:
        print(f"Error: No NRRD files found in {input_dir}")
        sys.exit(1)
    
    # 按文件名排序
    nrrd_files.sort()
    
    for mask_path in nrrd_files:
        filename = os.path.basename(mask_path)
        print(f"# processing {filename}")
        
        try:
            mask, _ = nrrd.read(mask_path)
            
            # 获取 mask 内所有体素的坐标（mask > 0）
            x_idx, y_idx, z_idx = np.where(mask > 0)
            
            if len(x_idx) == 0:
                print(f"Warning: {filename} has no non-zero values")
                continue
            
            # 计算 mask 的重心（质心）
            center_x = np.mean(x_idx)
            center_y = np.mean(y_idx)
            center_z = np.mean(z_idx)
            
            # 计算 crop 的边界（以重心为中心）
            half_x = bbox_x / 2.0
            half_y = bbox_y / 2.0
            half_z = bbox_z / 2.0
            
            crop_x_min = center_x - half_x
            crop_x_max = center_x + half_x
            crop_y_min = center_y - half_y
            crop_y_max = center_y + half_y
            crop_z_min = center_z - half_z
            crop_z_max = center_z + half_z
            
            # 统计在 crop 范围外的体素
            # 体素在 crop 外：x < crop_x_min 或 x >= crop_x_max，y 和 z 同理
            outside_mask = (
                (x_idx < crop_x_min) | (x_idx >= crop_x_max) |
                (y_idx < crop_y_min) | (y_idx >= crop_y_max) |
                (z_idx < crop_z_min) | (z_idx >= crop_z_max)
            )
            
            total_voxels = len(x_idx)
            outside_voxels = np.sum(outside_mask)
            outside_percentage = (outside_voxels / total_voxels * 100) if total_voxels > 0 else 0.0
            
            records.append({
                "filename": filename,
                "total_voxels": total_voxels,
                "outside_voxels": outside_voxels,
                "outside_percentage": outside_percentage
            })
            
            print(f"  Total voxels: {total_voxels}, Outside: {outside_voxels} ({outside_percentage:.2f}%)")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
        
    df = pd.DataFrame(records)
    
    # 打印汇总统计
    print(f"\n{'='*60}")
    print(f"Summary for {len(df)} files:")
    print(f"{'='*60}")
    print(f"Total voxels (mean): {df['total_voxels'].mean():.2f}")
    print(f"Outside voxels (mean): {df['outside_voxels'].mean():.2f}")
    print(f"Outside percentage (mean): {df['outside_percentage'].mean():.2f}%")
    print(f"Outside percentage (median): {df['outside_percentage'].median():.2f}%")
    print(f"Outside percentage (max): {df['outside_percentage'].max():.2f}%")
    print(f"Outside percentage (min): {df['outside_percentage'].min():.2f}%")
    
    print(f"\nProcessed {len(df)} files")
    print(df.head())
    
    # 保存到 CSV（如果指定了输出文件）
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bounding box analysis tool for NRRD files')
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # 功能一：分析 nrrd 文件
    parser_analyze = subparsers.add_parser('analyze', help='Analyze bbox distances from NRRD files in input directory')
    parser_analyze.add_argument('input_dir', type=str, help='Input directory containing NRRD files')
    parser_analyze.add_argument('output_csv', type=str, help='Output CSV file path')
    
    # 功能二：计算百分位数
    parser_percentile = subparsers.add_parser('percentile', help='Calculate percentiles from CSV file')
    parser_percentile.add_argument('csv_file', type=str, help='Input CSV file path')
    
    # 功能三：统计 crop 外的体素
    parser_crop = subparsers.add_parser('crop', help='Analyze voxels outside 3D crop bbox centered at mask centroid')
    parser_crop.add_argument('input_dir', type=str, help='Input directory containing NRRD files')
    parser_crop.add_argument('bbox_size', type=str, help='Bounding box size in format "x0,y0,z0" (e.g., "64,64,64")')
    parser_crop.add_argument('--output_csv', type=str, default=None, help='Optional output CSV file path')
    
    args = parser.parse_args()
    
    if args.mode == 'analyze':
        if not os.path.isdir(args.input_dir):
            print(f"Error: Input directory does not exist: {args.input_dir}")
            sys.exit(1)
        analyze_bbox_from_nrrd(args.input_dir, args.output_csv)
    elif args.mode == 'percentile':
        calculate_percentiles(args.csv_file)
    elif args.mode == 'crop':
        if not os.path.isdir(args.input_dir):
            print(f"Error: Input directory does not exist: {args.input_dir}")
            sys.exit(1)
        analyze_crop_outside_voxels(args.input_dir, args.bbox_size, args.output_csv)
    else:
        parser.print_help()
        sys.exit(1)