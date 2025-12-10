#!/usr/bin/env python3
import nrrd
import numpy as np
import sys
import os
import argparse

def hu_clip_normalize(input_file, output_dir):
    data, header = nrrd.read(input_file)
    data_clipped = np.clip(data, -1000, 400)
    data_normalized = (data_clipped + 1000) / 1400
    
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(input_file)
    output_path = os.path.join(output_dir, filename)

    nrrd.write(output_path, data_normalized, header)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HU clip and min-max normalization for NRRD files')
    parser.add_argument('input_file', type=str, help='Input NRRD file path')
    parser.add_argument('output_dir', type=str, help='Output directory path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file does not exist: {args.input_file}")
        sys.exit(1)
    
    hu_clip_normalize(args.input_file, args.output_dir)

