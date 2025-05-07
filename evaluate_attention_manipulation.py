#!/usr/bin/env python3
"""
Evaluate TACO compression with modified attention layer behavior on Kodak dataset.
This script tests the importance of attention patterns by manipulating attention during inference.
"""
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import lpips
from PIL import Image
from pytorch_msssim import ms_ssim
from pathlib import Path
from transformers import CLIPTextModel, AutoTokenizer

from config.config import model_config
from models import TACO
from utils.utils import *
from modules.transform.analysis import Injector, Extractor

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize LPIPS for evaluation
loss_fn_alex = lpips.LPIPS(net='alex')
loss_fn_alex = loss_fn_alex.to(device)
loss_fn_alex.requires_grad_(False)

def compute_psnr(a, b):
    """Calculate Peak Signal-to-Noise Ratio between two images"""
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)

# Attention manipulation hooks
def constant_attention_hook(module, input, output):
    """Replace attention outputs with constant uniform attention"""
    if isinstance(output, tuple) and len(output) == 2:
        attn_out, attn_weights = output
        # Replace with uniform attention (all tokens get equal attention)
        uniform_attn = torch.ones_like(attn_weights) / attn_weights.size(-1)
        return attn_out, uniform_attn
    return output

def zero_attention_hook(module, input, output):
    """Replace attention outputs with zeros (effectively disabling attention)"""
    if isinstance(output, tuple) and len(output) == 2:
        attn_out, attn_weights = output
        # Replace with zero attention
        zero_attn = torch.zeros_like(attn_out)
        return zero_attn, attn_weights
    return output

def random_attention_hook(module, input, output):
    """Replace attention outputs with random values"""
    if isinstance(output, tuple) and len(output) == 2:
        attn_out, attn_weights = output
        # Replace with random attention
        random_attn = torch.rand_like(attn_weights)
        # Normalize to make it a proper attention distribution
        random_attn = random_attn / random_attn.sum(dim=-1, keepdim=True)
        return attn_out, random_attn
    return output

def null_hook(module, input, output):
    """Do nothing, just return the original output (for baseline)"""
    return output

def set_gamma_to_zero(model):
    """Set gamma parameter to zero in all Injector modules"""
    for name, module in model.named_modules():
        if isinstance(module, Injector):
            logger.info(f"Setting gamma to zero for {name}")
            module.gamma.data.zero_()
    return model

def apply_attention_hooks(model, hook_type="baseline"):
    """Apply hooks to attention modules based on the specified type"""
    hooks = []
    
    if hook_type == "baseline":
        # No hooks for baseline
        logger.info("Running baseline model (no hooks)")
        return hooks
    
    # Count how many hooks were applied
    hook_count = 0
    
    for name, module in model.named_modules():
        # Target attention layers inside Injector/Extractor modules
        if isinstance(module, nn.MultiheadAttention):
            if hook_type == "constant":
                hook = module.register_forward_hook(constant_attention_hook)
                hook_count += 1
            elif hook_type == "zero":
                hook = module.register_forward_hook(zero_attention_hook)
                hook_count += 1
            elif hook_type == "random":
                hook = module.register_forward_hook(random_attention_hook)
                hook_count += 1
            else:
                continue
                
            hooks.append(hook)
            logger.info(f"Applied {hook_type} hook to {name}")
    
    logger.info(f"Applied {hook_count} {hook_type} hooks in total")
    return hooks

def modify_injector_forward(model, modification_type="none"):
    """Modify Injector forward method to bypass or alter attention"""
    if modification_type == "none":
        return model
        
    # Store original forward methods to restore later
    original_forwards = {}
    
    for name, module in model.named_modules():
        if isinstance(module, Injector):
            # Save original forward method
            original_forwards[name] = module.forward
            
            if modification_type == "bypass":
                # Define new forward that bypasses attention
                def bypass_forward(self, image_features, text_features):
                    # Skip attention calculation and return image features unchanged
                    b, c, h, w = image_features.size()
                    return image_features
                
                # Bind the new method to the module
                import types
                module.forward = types.MethodType(bypass_forward, module)
                logger.info(f"Bypassed attention in {name}")
    
    return model, original_forwards

def restore_original_forwards(model, original_forwards):
    """Restore original forward methods"""
    for name, forward_method in original_forwards.items():
        for module_name, module in model.named_modules():
            if module_name == name:
                module.forward = forward_method
                break
    return model

def process_kodak_dataset(model, tokenizer, clip_model, manipulation_type="baseline", checkpoint_name=""):
    """Process Kodak dataset and evaluate with different attention manipulations"""
    kodak_dir = "./kodak"
    results_dir = f"./attention_manipulation_results/{checkpoint_name}/{manipulation_type}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load all Kodak images
    image_files = [f for f in os.listdir(kodak_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    
    # Track metrics
    avg_psnr = 0.0
    avg_ms_ssim = 0.0
    avg_lpips = 0.0
    avg_bpp = 0.0
    
    # Just use one generic caption since we're testing attention manipulation, not caption impact
    test_captions = ["a detailed photograph"]
    
    # Process each image with different captions
    for caption in test_captions:
        caption_results = {
            'psnr': [],
            'ms_ssim': [],
            'lpips': [],
            'bpp': []
        }
        
        # Create caption-specific directory
        caption_safe = caption.replace(" ", "_")[:20] if caption else "empty_caption"
        caption_dir = os.path.join(results_dir, caption_safe)
        os.makedirs(caption_dir, exist_ok=True)
        
        logger.info(f"Testing with caption: '{caption}'")
        
        for img_name in image_files:
            img_path = os.path.join(kodak_dir, img_name)
            output_path = os.path.join(caption_dir, img_name)
            
            # Load and prepare image
            img = Image.open(img_path).convert('RGB')
            x = torchvision.transforms.ToTensor()(img).unsqueeze(0).to(device)
            
            # Process image dimensions
            _, _, H, W = x.shape
            pad_h = 0
            pad_w = 0
            if H % 64 != 0:
                pad_h = 64 * (H // 64 + 1) - H
            if W % 64 != 0:
                pad_w = 64 * (W // 64 + 1) - W
            
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
            
            # Process caption
            if caption.strip():
                tokens = tokenizer(caption, return_tensors="pt").to(device)
                with torch.no_grad():
                    text_embeddings = clip_model(**tokens).last_hidden_state
            else:
                # Empty caption - zero embedding
                text_embeddings = torch.zeros(1, 1, 768, device=device)  # CLIP embedding dimension
            
            # Compress and decompress
            with torch.no_grad():
                out_enc = model.compress(x_padded, text_embeddings)
                shape = out_enc["strings"]
                
                # Calculate bpp
                num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
                bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                
                # Decompress
                out_dec = model.decompress(out_enc["strings"], out_enc["shape"], text_embeddings)
                x_hat = out_dec["x_hat"]
                
                # Remove padding
                x_hat = x_hat[:, :, 0:H, 0:W]
            
            # Calculate metrics
            psnr = compute_psnr(x, x_hat)
            ms_ssim_val = ms_ssim(x, x_hat, data_range=1.0).item()
            lpips_val = loss_fn_alex(x, x_hat).item()
            
            # Save results for this image
            caption_results['psnr'].append(psnr)
            caption_results['ms_ssim'].append(ms_ssim_val)
            caption_results['lpips'].append(lpips_val)
            caption_results['bpp'].append(bpp)
            
            # Log and save reconstructed image
            logger.info(f"Image: {img_name}, Caption: '{caption}', PSNR: {psnr:.2f}, MS-SSIM: {ms_ssim_val:.4f}, LPIPS: {lpips_val:.4f}, BPP: {bpp:.4f}")
            torchvision.utils.save_image(x_hat, output_path)
        
        # Calculate averages for this caption
        avg_caption_psnr = sum(caption_results['psnr']) / len(caption_results['psnr'])
        avg_caption_ms_ssim = sum(caption_results['ms_ssim']) / len(caption_results['ms_ssim'])
        avg_caption_lpips = sum(caption_results['lpips']) / len(caption_results['lpips'])
        avg_caption_bpp = sum(caption_results['bpp']) / len(caption_results['bpp'])
        
        logger.info(f"Average for caption '{caption}':")
        logger.info(f"PSNR: {avg_caption_psnr:.2f}, MS-SSIM: {avg_caption_ms_ssim:.4f}, LPIPS: {avg_caption_lpips:.4f}, BPP: {avg_caption_bpp:.4f}")
        
        # Add to overall averages
        avg_psnr += avg_caption_psnr
        avg_ms_ssim += avg_caption_ms_ssim
        avg_lpips += avg_caption_lpips
        avg_bpp += avg_caption_bpp
        
        # Save caption results to CSV
        with open(os.path.join(caption_dir, "metrics.csv"), "w") as f:
            f.write("Image,PSNR,MS-SSIM,LPIPS,BPP\n")
            for i, img_name in enumerate(image_files):
                f.write(f"{img_name},{caption_results['psnr'][i]:.4f},{caption_results['ms_ssim'][i]:.4f},{caption_results['lpips'][i]:.4f},{caption_results['bpp'][i]:.4f}\n")
    
    # Calculate overall averages
    avg_psnr /= len(test_captions)
    avg_ms_ssim /= len(test_captions)
    avg_lpips /= len(test_captions)
    avg_bpp /= len(test_captions)
    
    # Save overall results
    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write(f"Manipulation: {manipulation_type}\n")
        f.write(f"Average PSNR: {avg_psnr:.4f}\n")
        f.write(f"Average MS-SSIM: {avg_ms_ssim:.4f}\n")
        f.write(f"Average LPIPS: {avg_lpips:.4f}\n")
        f.write(f"Average BPP: {avg_bpp:.4f}\n")
    
    return {
        'checkpoint': checkpoint_name,
        'manipulation': manipulation_type,
        'psnr': avg_psnr,
        'ms_ssim': avg_ms_ssim,
        'lpips': avg_lpips,
        'bpp': avg_bpp
    }

def main():
    # Add import for glob and csv
    import glob
    import csv
    
    # Define manipulation types
    manipulations = ["baseline", "gamma_zero", "constant", "random", "bypass"]
    
    # Load CLIP model and tokenizer
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPTextModel.from_pretrained(clip_model_name).to(device)
    clip_model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
    
    # Load model configuration
    taco_config = model_config()
    
    # Find all checkpoint files
    checkpoint_dir = "./checkpoint"
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "lambda_*.pth.tar")))
    
    if not checkpoint_files:
        logger.error("No checkpoint files found in ./checkpoint directory!")
        return
    
    # All results
    all_results = []
    
    # Create results directory
    os.makedirs("./attention_manipulation_results", exist_ok=True)
    
    # Process each checkpoint
    for checkpoint_file in checkpoint_files:
        checkpoint_name = os.path.basename(checkpoint_file).replace(".pth.tar", "")
        logger.info(f"\n==== Processing checkpoint {checkpoint_name} ====\n")
        
        # Load model
        model = TACO(taco_config, text_embedding_dim=clip_model.config.hidden_size)
        checkpoint = torch.load(checkpoint_file, map_location=device)
        
        # Load state dict
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except:
            # Try without module prefix
            new_state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                new_state_dict[k.replace("module.", "")] = v
            model.load_state_dict(new_state_dict)
        
        model = model.to(device)
        model.eval()
        model.update()
        
        # Process with different attention manipulations
        for manipulation in manipulations:
            logger.info(f"\n--- Running {checkpoint_name} with {manipulation} manipulation ---\n")
            
            # Create a fresh copy of the model for each manipulation
            eval_model = TACO(taco_config, text_embedding_dim=clip_model.config.hidden_size)
            eval_model.load_state_dict(model.state_dict())
            eval_model = eval_model.to(device)
            eval_model.eval()
            eval_model.update()
            
            hooks = []
            original_forwards = {}
            
            # Apply the selected manipulation
            if manipulation == "baseline":
                # No modifications for baseline
                pass
            elif manipulation == "gamma_zero":
                eval_model = set_gamma_to_zero(eval_model)
            elif manipulation == "bypass":
                eval_model, original_forwards = modify_injector_forward(eval_model, "bypass")
            else:
                # Apply hooks for constant, random, etc.
                hooks = apply_attention_hooks(eval_model, manipulation)
            
            # Run evaluation
            results = process_kodak_dataset(eval_model, tokenizer, clip_model, manipulation, checkpoint_name)
            all_results.append(results)
            
            # Clean up hooks
            for hook in hooks:
                hook.remove()
            
            # Restore original forwards if modified
            if original_forwards:
                eval_model = restore_original_forwards(eval_model, original_forwards)
            
            # Clear memory
            del eval_model
            torch.cuda.empty_cache()
        
        # Clean up main model
        del model
        torch.cuda.empty_cache()
    
    # Write overall comparison to CSV
    with open("./attention_manipulation_results/all_results.csv", "w", newline='') as f:
        fieldnames = ['checkpoint', 'manipulation', 'psnr', 'ms_ssim', 'lpips', 'bpp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)
    
    # Log final summary
    logger.info("\n==== Overall Results Summary ====\n")
    logger.info(f"{'Checkpoint':<15} {'Manipulation':<12} {'PSNR':<10} {'MS-SSIM':<10} {'LPIPS':<10} {'BPP':<10}")
    logger.info("-" * 70)
    
    for result in all_results:
        logger.info(
            f"{result['checkpoint']:<15} {result['manipulation']:<12} "
            f"{result['psnr']:<10.4f} {result['ms_ssim']:<10.4f} "
            f"{result['lpips']:<10.4f} {result['bpp']:<10.4f}"
        )

if __name__ == "__main__":
    main()
