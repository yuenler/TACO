#!/usr/bin/env python3
"""
Evaluate ELIC compression across multiple checkpoints on the Kodak dataset.
Adapted from https://github.com/VincentChandelier/ELiC-ReImplemetation/blob/main/Inference.py
"""
import json
import sys
import time
import csv
import os
import glob
import math
from pathlib import Path
from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import lpips
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms
from tqdm import tqdm
import compressai
from compressai.zoo import load_state_dict

from Network import TestModel

# Set deterministic behavior for reproducibility
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# Image extensions to consider
IMG_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"
)

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize LPIPS
loss_fn_alex = lpips.LPIPS(net='alex')
loss_fn_alex = loss_fn_alex.to(device)
loss_fn_alex.requires_grad_(False)

def compute_psnr(a, b):
    """Calculate Peak Signal-to-Noise Ratio between two images"""
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def collect_images(rootpath: str) -> List[str]:
    """Collect all images from a directory"""
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]

def read_image(filepath: str) -> torch.Tensor:
    """Read an image file and convert to tensor"""
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)

def load_model(checkpoint_path):
    """
    Load the ELIC model with specified checkpoint
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = load_state_dict(torch.load(checkpoint_path))
    model = TestModel().from_state_dict(state_dict).eval()
    model = model.to(device)
    return model

@torch.no_grad()
def compress_and_evaluate(model, image, patch=64):
    """
    Compress image using provided model, then decompress and calculate PSNR and LPIPS
    """
    # Prepare image
    x = image.unsqueeze(0)
    h, w = x.size(2), x.size(3)
    
    # Original padding method from ELIC
    p = patch  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = 0
    padding_right = new_w - w - padding_left
    padding_top = 0
    padding_bottom = new_h - h - padding_top
    pad = nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 0)
    x_padded = pad(x)

    # Compress
    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    # Decompress
    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    # Remove padding
    out_dec["x_hat"] = torch.nn.functional.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    # Calculate BPP
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = 0
    for s in out_enc["strings"]:
        for j in s:
            if isinstance(j, list):
                for i in j:
                    if isinstance(i, list):
                        for k in i:
                            bpp += len(k)
                    else:
                        bpp += len(i)
            else:
                bpp += len(j)
    bpp *= 8.0 / num_pixels
    
    # Calculate PSNR
    psnr_value = compute_psnr(x, out_dec["x_hat"])
    
    # Calculate LPIPS
    lpips_value = loss_fn_alex(x, out_dec["x_hat"]).item()
    
    return {
        "psnr": psnr_value,
        "lpips": lpips_value,
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
        "x_hat": out_dec["x_hat"]  # Return reconstructed image for saving if needed
    }

def evaluate_checkpoint(model, image_files, patch=64):
    """
    Evaluate a single checkpoint on the dataset
    """
    results = {
        "psnr": [],
        "lpips": [],
        "bpp": [],
        "encoding_time": [],
        "decoding_time": []
    }
    
    # Process each image in the dataset
    for image_file in tqdm(image_files, desc=f"Processing images"):
        # Load and process image
        x = read_image(image_file).to(device)
        
        # Compress and evaluate
        metrics = compress_and_evaluate(model, x, patch=patch)
        
        # Store results
        results["psnr"].append(metrics["psnr"])
        results["lpips"].append(metrics["lpips"])
        results["bpp"].append(metrics["bpp"])
        results["encoding_time"].append(metrics["encoding_time"])
        results["decoding_time"].append(metrics["decoding_time"])
    
    # Calculate averages
    avg_results = {
        "avg_psnr": float(np.mean(results["psnr"])),
        "std_psnr": float(np.std(results["psnr"])),
        "avg_lpips": float(np.mean(results["lpips"])),
        "std_lpips": float(np.std(results["lpips"])),
        "avg_bpp": float(np.mean(results["bpp"])),
        "std_bpp": float(np.std(results["bpp"])),
        "avg_encoding_time": float(np.mean(results["encoding_time"])),
        "avg_decoding_time": float(np.mean(results["decoding_time"]))
    }
    
    return avg_results

def main():
    # Hardcoded parameters (similar to TACO evaluation script)
    kodak_dir = "./kodak"
    checkpoint_dir = "./elic_checkpoints"  # Directory containing ELIC checkpoints
    output_file = "elic_kodak_results.json"  # Output JSON file
    patch_size = 64  # Patch size for padding
    
    # Set entropy coder
    compressai.set_entropy_coder("ans")
    
    # Get list of image files
    image_files = collect_images(kodak_dir)
    if len(image_files) == 0:
        print(f"Error: no images found in directory {kodak_dir}", file=sys.stderr)
        sys.exit(1)
    image_files = sorted(image_files)
    print(f"Found {len(image_files)} images in Kodak dataset")
    
    # Get list of checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pth.tar"))
    if len(checkpoint_files) == 0:
        # Try .pt files if no .pth.tar files found
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        
    if len(checkpoint_files) == 0:
        print(f"Error: no checkpoint files found in directory {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)
    checkpoint_files = sorted(checkpoint_files)
    print(f"Using checkpoints: {[os.path.basename(cp) for cp in checkpoint_files]}")
    
    # Evaluate each checkpoint
    all_results = {}
    
    for checkpoint_path in checkpoint_files:
        checkpoint_name = os.path.basename(checkpoint_path)
        print(f"\nEvaluating checkpoint: {checkpoint_name}")
        
        # Load model
        model = load_model(checkpoint_path)
        
        # Evaluate checkpoint
        results = evaluate_checkpoint(model, image_files, patch_size)
        all_results[checkpoint_name] = results
        
        print(f"Results for {checkpoint_name}:")
        print(f"  PSNR={results['avg_psnr']:.4f}, LPIPS={results['avg_lpips']:.4f}, BPP={results['avg_bpp']:.4f}")
    
    # Save results to JSON
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nSaved detailed results to {output_file}")
    
    # Print numerical results summary
    print("\nNumerical Results:")
    print("-" * 100)
    print(f"{'Checkpoint':<25} {'PSNR':<10} {'LPIPS':<10} {'BPP':<10}")
    print("-" * 100)
    
    for cp_name, results in all_results.items():
        print(f"{cp_name:<25} {results['avg_psnr']:.4f} {results['avg_lpips']:.4f} {results['avg_bpp']:.4f}")
        print("-" * 100)
    
    print("\nTo generate plots comparing ELIC with TACO, run the plotting script.")

if __name__ == "__main__":
    main()
