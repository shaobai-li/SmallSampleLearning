#!/usr/bin/env python3
import nrrd
import numpy as np
import sys
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply mask and perform 3D crop on NRRD files')
    parser.add_argument('image_nrrd', type=str, help='Input image NRRD file path')
    parser.add_argument('mask_nrrd', type=str, help='Input mask NRRD file path')
    parser.add_argument('bbox_size', type=str, help='Bounding box size in format "x,y,z" (e.g., "64,64,64")')
    parser.add_argument('output_dir', type=str, help='Output directory path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_nrrd):
        print(f"Error: Image file does not exist: {args.image_nrrd}")
        sys.exit(1)
    
    if not os.path.exists(args.mask_nrrd):
        print(f"Error: Mask file does not exist: {args.mask_nrrd}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        bbox_x, bbox_y, bbox_z = map(int, args.bbox_size.split(','))
        
        image, image_header = nrrd.read(args.image_nrrd)
        mask, _ = nrrd.read(args.mask_nrrd)
        
        if image.shape != mask.shape:
            raise ValueError(f"Image shape {image.shape} does not match mask shape {mask.shape}")
        
        masked_image = image * (mask > 0).astype(image.dtype)
        
        x_idx, y_idx, z_idx = np.where(mask > 0)
        if len(x_idx) == 0:
            raise ValueError("Mask has no non-zero values")
        
        center_x = np.mean(x_idx)
        center_y = np.mean(y_idx)
        center_z = np.mean(z_idx)
        
        half_x = bbox_x / 2.0
        half_y = bbox_y / 2.0
        half_z = bbox_z / 2.0
        
        x_min = int(np.floor(center_x - half_x))
        x_max = int(np.ceil(center_x + half_x))
        y_min = int(np.floor(center_y - half_y))
        y_max = int(np.ceil(center_y + half_y))
        z_min = int(np.floor(center_z - half_z))
        z_max = int(np.ceil(center_z + half_z))
        
        img_z, img_y, img_x = image.shape
        
        pad_x_before = max(0, -x_min)
        pad_x_after = max(0, x_max - img_x)
        pad_y_before = max(0, -y_min)
        pad_y_after = max(0, y_max - img_y)
        pad_z_before = max(0, -z_min)
        pad_z_after = max(0, z_max - img_z)
        
        if pad_x_before > 0 or pad_x_after > 0 or pad_y_before > 0 or pad_y_after > 0 or pad_z_before > 0 or pad_z_after > 0:
            pad_width = ((pad_z_before, pad_z_after), (pad_y_before, pad_y_after), (pad_x_before, pad_x_after))
            masked_image = np.pad(masked_image, pad_width, mode='constant', constant_values=0)
            x_min += pad_x_before
            x_max += pad_x_before
            y_min += pad_y_before
            y_max += pad_y_before
            z_min += pad_z_before
            z_max += pad_z_before
        
        cropped_image = masked_image[z_min:z_max, y_min:y_max, x_min:x_max]
        
        filename = os.path.basename(args.image_nrrd)
        output_path = os.path.join(args.output_dir, filename)
        
        nrrd.write(output_path, cropped_image, image_header)
        print(f"Saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
