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
    
    args = parser.parse_args()
    
    if args.mode == 'analyze':
        if not os.path.isdir(args.input_dir):
            print(f"Error: Input directory does not exist: {args.input_dir}")
            sys.exit(1)
        analyze_bbox_from_nrrd(args.input_dir, args.output_csv)
    elif args.mode == 'percentile':
        calculate_percentiles(args.csv_file)
    else:
        parser.print_help()
        sys.exit(1)