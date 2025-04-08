#!/usr/bin/env python3
import os
import torch
import torchvision
import json
import random
import numpy as np
import lpips
import torch.nn.functional as F
from pathlib import Path
from transformers import CLIPTextModel, AutoTokenizer
from tqdm import tqdm

from config.config import model_config
from models import TACO
from utils.utils import *

"""
This script evaluates TACO compression with different caption scenarios:
1. Correct caption
2. No caption
3. Random caption from another image

It selects 20 random images from the COCO validation dataset and computes
the average LPIPS score for each scenario.
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

def main():
    # Parameters
    coco_dir = "./coco"
    val_images_dir = os.path.join(coco_dir, "val2014")
    captions_file = os.path.join(coco_dir, "val2014_captions.json")
    checkpoint_path = "checkpoint/lambda_0.004.pth.tar"  # Default checkpoint
    num_samples = 20
    
    # Load captions
    print(f"Loading captions from {captions_file}")
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)
    
    # Get list of image files and their captions
    image_files = list(captions_data.keys())
    print(f"Found {len(image_files)} images with captions")
    
    # Select random samples
    if len(image_files) < num_samples:
        print(f"Warning: Only {len(image_files)} images available, using all of them")
        samples = image_files
    else:
        samples = random.sample(image_files, num_samples)
    
    # Load model
    net, text_model, tokenizer = load_model(checkpoint_path)
    
    # Results containers
    results = {
        "correct_caption": [],
        "no_caption": [],
        "random_caption": []
    }
    
    bpp_results = {
        "correct_caption": [],
        "no_caption": [],
        "random_caption": []
    }
    
    # Process each sample
    for image_file in tqdm(samples, desc="Processing images"):
        image_path = os.path.join(val_images_dir, image_file)
        correct_caption = captions_data[image_file]
        
        # Get random caption (ensure it's different from correct one)
        random_image = image_file
        while random_image == image_file:
            random_image = random.choice(image_files)
        random_caption = captions_data[random_image]
        
        # Load and process image
        x, x_padded, original_size = process_image(image_path)
        
        # 1. Compress with correct caption
        text_embeddings = process_caption(correct_caption, tokenizer, text_model)
        lpips_score, bpp = compress_and_evaluate(net, x, x_padded, original_size, text_embeddings)
        results["correct_caption"].append(lpips_score)
        bpp_results["correct_caption"].append(bpp)
        
        # 2. Compress with no caption
        text_embeddings = process_caption("", tokenizer, text_model)
        lpips_score, bpp = compress_and_evaluate(net, x, x_padded, original_size, text_embeddings)
        results["no_caption"].append(lpips_score)
        bpp_results["no_caption"].append(bpp)
        
        # 3. Compress with random caption
        text_embeddings = process_caption(random_caption, tokenizer, text_model)
        lpips_score, bpp = compress_and_evaluate(net, x, x_padded, original_size, text_embeddings)
        results["random_caption"].append(lpips_score)
        bpp_results["random_caption"].append(bpp)
    
    # Print results
    print("\nResults summary:")
    print(f"Number of samples: {len(samples)}")
    print("\nAverage LPIPS scores (lower is better):")
    for scenario, scores in results.items():
        print(f"  {scenario.replace('_', ' ').title()}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    print("\nAverage BPP (bits per pixel):")
    for scenario, scores in bpp_results.items():
        print(f"  {scenario.replace('_', ' ').title()}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # Create detailed results file
    results_data = {
        "samples": samples,
        "individual_results": {
            "correct_caption": {samples[i]: {"lpips": results["correct_caption"][i], "bpp": bpp_results["correct_caption"][i]} for i in range(len(samples))},
            "no_caption": {samples[i]: {"lpips": results["no_caption"][i], "bpp": bpp_results["no_caption"][i]} for i in range(len(samples))},
            "random_caption": {samples[i]: {"lpips": results["random_caption"][i], "bpp": bpp_results["random_caption"][i]} for i in range(len(samples))}
        },
        "average_results": {
            "correct_caption": {"lpips_mean": float(np.mean(results["correct_caption"])), "lpips_std": float(np.std(results["correct_caption"])), 
                               "bpp_mean": float(np.mean(bpp_results["correct_caption"])), "bpp_std": float(np.std(bpp_results["correct_caption"]))},
            "no_caption": {"lpips_mean": float(np.mean(results["no_caption"])), "lpips_std": float(np.std(results["no_caption"])),
                          "bpp_mean": float(np.mean(bpp_results["no_caption"])), "bpp_std": float(np.std(bpp_results["no_caption"]))},
            "random_caption": {"lpips_mean": float(np.mean(results["random_caption"])), "lpips_std": float(np.std(results["random_caption"])),
                              "bpp_mean": float(np.mean(bpp_results["random_caption"])), "bpp_std": float(np.std(bpp_results["random_caption"]))}
        }
    }
    
    with open("caption_impact_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    
    print("\nDetailed results saved to caption_impact_results.json")

if __name__ == "__main__":
    main()
