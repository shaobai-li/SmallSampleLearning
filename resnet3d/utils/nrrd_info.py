import nrrd
import numpy as np
import sys
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display information about NRRD files')
    parser.add_argument('input_file', type=str, help='Input NRRD file path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file does not exist: {args.input_file}")
        sys.exit(1)
    
    data, header = nrrd.read(args.input_file)
    print(f"shape={data.shape}, min={data.min()}, max={data.max()}")
