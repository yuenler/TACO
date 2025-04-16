#!/usr/bin/env python3
import os
import torch
import torchvision
import json
import numpy as np
import lpips
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import CLIPTextModel, AutoTokenizer
from tqdm import tqdm
import glob

from config.config import model_config
from models import TACO
from utils.utils import *

"""
This script evaluates TACO compression across multiple checkpoints on the Kodak dataset:
1. With caption (using OFA captions)
2. Without caption (empty string)

For each checkpoint, it computes average LPIPS and BPP metrics and generates
a graph plotting LPIPS vs BPP for both scenarios.
"""

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize LPIPS
loss_fn_alex = lpips.LPIPS(net='alex')
loss_fn_alex = loss_fn_alex.to(device)
loss_fn_alex.requires_grad_(False)

def load_model(checkpoint_path):
    """
    Load the TACO model with specified checkpoint
    """
    # Load CLIP text model
    clip_model_name = "openai/clip-vit-base-patch32"
    CLIP_text_model = CLIPTextModel.from_pretrained(clip_model_name).to(device)
    CLIP_text_model.requires_grad_(False)
    CLIP_tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
    
    # Load TACO model
    taco_config = model_config()
    net = TACO(taco_config, text_embedding_dim=CLIP_text_model.config.hidden_size)
    net = net.eval().to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)['state_dict']
    
    # Handle different state dict formats
    try:
        net.load_state_dict(state_dict)
    except:
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        net.load_state_dict(new_state_dict)
    
    net.requires_grad_(False)
    net.update()
    
    return net, CLIP_text_model, CLIP_tokenizer

def process_image(image_path):
    """
    Load and process image for the model
    """
    image = torchvision.io.read_image(image_path).float() / 255.0
    image = image.to(device)
    
    # Make sure it's 3-channel
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] > 3:
        image = image[:3]
    
    # Original image dimensions
    _, H, W = image.shape
    x = image.unsqueeze(0)
    
    # Pad if necessary
    pad_h, pad_w = 0, 0
    if H % 64 != 0:
        pad_h = 64 * (H // 64 + 1) - H
    if W % 64 != 0:
        pad_w = 64 * (W // 64 + 1) - W
    
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
    
    return x, x_padded, (H, W)

def process_caption(caption, tokenizer, text_model):
    """
    Process caption for the model
    """
    clip_token = tokenizer([caption], padding="max_length", max_length=38, truncation=True, return_tensors="pt").to(device)
    text_embeddings = text_model(**clip_token).last_hidden_state
    return text_embeddings

def compress_and_evaluate(net, image, x_padded, original_size, text_embeddings):
    """
    Compress image using provided model and text embeddings,
    then decompress and calculate LPIPS score
    """
    # Compress
    out_enc = net.compress(x_padded, text_embeddings)
    shape = out_enc["shape"]
    
    # Save compressed file to temporary file
    output_file = "temp_compressed.bin"
    with Path(output_file).open("wb") as f:
        write_uints(f, original_size)
        write_body(f, shape, out_enc["strings"])
    
    # Calculate BPP
    size = filesize(output_file)
    bpp = float(size) * 8 / (original_size[0] * original_size[1])
    
    # Decompress
    with Path(output_file).open("rb") as f:
        original_size_read = read_uints(f, 2)
        strings, shape = read_body(f)
    
    out = net.decompress(strings, shape, text_embeddings)
    x_hat = out["x_hat"].detach().clone()
    x_hat = x_hat[:, :, 0:original_size[0], 0:original_size[1]]
    
    # Calculate LPIPS
    lpips_score = loss_fn_alex(image, x_hat).item()
    
    # Clean up
    if os.path.exists(output_file):
        os.remove(output_file)
    
    return lpips_score, bpp

def evaluate_checkpoint(checkpoint_path, image_files, captions_data, kodak_dir):
    """
    Evaluate a single checkpoint with and without captions
    """
    # Load model
    net, text_model, tokenizer = load_model(checkpoint_path)
    
    # Results containers
    results = {
        "with_caption": {"lpips": [], "bpp": []},
        "no_caption": {"lpips": [], "bpp": []}
    }
    
    # Process each image in the Kodak dataset
    for image_file in tqdm(image_files, desc=f"Processing images for {os.path.basename(checkpoint_path)}"):
        image_path = os.path.join(kodak_dir, image_file)
        correct_caption = captions_data[image_file]
        
        # Load and process image
        x, x_padded, original_size = process_image(image_path)
        
        # 1. Compress with correct caption
        text_embeddings = process_caption(correct_caption, tokenizer, text_model)
        lpips_score, bpp = compress_and_evaluate(net, x, x_padded, original_size, text_embeddings)
        results["with_caption"]["lpips"].append(lpips_score)
        results["with_caption"]["bpp"].append(bpp)
        
        # 2. Compress with no caption
        text_embeddings = process_caption("", tokenizer, text_model)
        lpips_score, bpp = compress_and_evaluate(net, x, x_padded, original_size, text_embeddings)
        results["no_caption"]["lpips"].append(lpips_score)
        results["no_caption"]["bpp"].append(bpp)
    
    # Calculate averages
    avg_results = {
        "with_caption": {
            "avg_lpips": float(np.mean(results["with_caption"]["lpips"])),
            "std_lpips": float(np.std(results["with_caption"]["lpips"])),
            "avg_bpp": float(np.mean(results["with_caption"]["bpp"])),
            "std_bpp": float(np.std(results["with_caption"]["bpp"]))
        },
        "no_caption": {
            "avg_lpips": float(np.mean(results["no_caption"]["lpips"])),
            "std_lpips": float(np.std(results["no_caption"]["lpips"])),
            "avg_bpp": float(np.mean(results["no_caption"]["bpp"])),
            "std_bpp": float(np.std(results["no_caption"]["bpp"]))
        }
    }
    
    return avg_results

def extract_lambda_value(checkpoint_path):
    """Extract the lambda value from checkpoint filename for sorting"""
    filename = os.path.basename(checkpoint_path)
    # Extract the numeric value between "lambda_" and ".pth.tar"
    lambda_str = filename.split('lambda_')[1].split('.pth.tar')[0]
    return float(lambda_str)

def main():
    # Parameters
    kodak_dir = "./kodak"
    captions_file = "./materials/kodak_ofa.json"
    
    # Define checkpoint paths manually based on the Docker setup
    # These will be available when mounted in Docker at /app/checkpoint
    lambda_values = [0.0008, 0.0016, 0.004, 0.009, 0.015]
    checkpoint_paths = [f"checkpoint/lambda_{lam}.pth.tar" for lam in lambda_values]
    
    print(f"Using checkpoints: {[os.path.basename(cp) for cp in checkpoint_paths]}")
    
    # Load captions
    print(f"Loading captions from {captions_file}")
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)
    
    # Get list of image files
    image_files = list(captions_data.keys())
    print(f"Found {len(image_files)} images with captions")
    
    # Evaluate each checkpoint
    all_results = {}
    
    for checkpoint_path in checkpoint_paths:
        checkpoint_name = os.path.basename(checkpoint_path)
        print(f"\nEvaluating checkpoint: {checkpoint_name}")
        
        results = evaluate_checkpoint(checkpoint_path, image_files, captions_data, kodak_dir)
        all_results[checkpoint_name] = results
        
        print(f"Results for {checkpoint_name}:")
        print(f"  With Caption: LPIPS={results['with_caption']['avg_lpips']:.4f}, BPP={results['with_caption']['avg_bpp']:.4f}")
        print(f"  No Caption: LPIPS={results['no_caption']['avg_lpips']:.4f}, BPP={results['no_caption']['avg_bpp']:.4f}")
    
    # Save results to JSON
    with open("kodak_checkpoint_comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\nSaved detailed results to kodak_checkpoint_comparison_results.json")
    
    # Generate plot
    plt.figure(figsize=(10, 6))
    
    # Data for plotting
    x_with_caption = [all_results[cp]['with_caption']['avg_bpp'] for cp in checkpoint_paths]
    y_with_caption = [all_results[cp]['with_caption']['avg_lpips'] for cp in checkpoint_paths]
    
    x_no_caption = [all_results[cp]['no_caption']['avg_bpp'] for cp in checkpoint_paths]
    y_no_caption = [all_results[cp]['no_caption']['avg_lpips'] for cp in checkpoint_paths]
    
    # Sort points by BPP for proper line drawing
    with_caption_points = sorted(zip(x_with_caption, y_with_caption))
    no_caption_points = sorted(zip(x_no_caption, y_no_caption))
    
    x_with_caption = [p[0] for p in with_caption_points]
    y_with_caption = [p[1] for p in with_caption_points]
    
    x_no_caption = [p[0] for p in no_caption_points]
    y_no_caption = [p[1] for p in no_caption_points]
    
    # Plot
    plt.plot(x_with_caption, y_with_caption, 'o-', color='green', linewidth=2, label='With Caption')
    plt.plot(x_no_caption, y_no_caption, 'o-', color='blue', linewidth=2, label='No Caption')
    
    # Add checkpoint labels
    for i, cp in enumerate(checkpoint_paths):
        lambda_val = extract_lambda_value(cp)
        plt.annotate(f"λ={lambda_val}", 
                    (all_results[os.path.basename(cp)]['with_caption']['avg_bpp'], 
                     all_results[os.path.basename(cp)]['with_caption']['avg_lpips']),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    # Styling
    plt.xlabel('Bits per pixel (BPP)')
    plt.ylabel('LPIPS (lower is better) ↓')
    plt.title('TACO Performance on Kodak Dataset: Caption Impact Across Checkpoints')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save plot
    plt.savefig('kodak_caption_impact_plot.png', dpi=300, bbox_inches='tight')
    print("Generated plot saved as kodak_caption_impact_plot.png")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    main()
