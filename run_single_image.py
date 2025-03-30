import os
import torch
import torchvision
import argparse
from pathlib import Path
import math
import lpips
import torch.nn.functional as F
from pytorch_msssim import ms_ssim as ms_ssim_func
from transformers import CLIPTextModel, AutoTokenizer

from config.config import model_config 
from models import TACO
from utils.utils import *

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize LPIPS
loss_fn_alex = lpips.LPIPS(net='alex')
loss_fn_alex = loss_fn_alex.to(device)
loss_fn_alex.requires_grad_(False)

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def parse_args():
    parser = argparse.ArgumentParser(description="Run TACO on a single image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to a single image")
    parser.add_argument("--caption", type=str, required=True, help="Caption for the image")
    parser.add_argument("--checkpoint", type=str, default="checkpoint/lambda_0.0004.pth.tar", help="Path to the model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save output")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)['state_dict']
    
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
    
    # Load and process image
    print(f"Processing image: {args.image_path}")
    image = torchvision.io.read_image(args.image_path).float() / 255.0
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
    
    # Process caption
    caption = args.caption
    print(f"Using caption: {caption}")
    clip_token = CLIP_tokenizer([caption], padding="max_length", max_length=38, truncation=True, return_tensors="pt").to(device)
    text_embeddings = CLIP_text_model(**clip_token).last_hidden_state
    
    # Compress
    print("Compressing image...")
    out_enc = net.compress(x_padded, text_embeddings)
    shape = out_enc["shape"]
    
    # Save compressed file
    output_file = os.path.join(args.output_dir, "compressed.bin")
    with Path(output_file).open("wb") as f:
        write_uints(f, (H, W))
        write_body(f, shape, out_enc["strings"])
    
    # Calculate BPP
    size = filesize(output_file)
    bpp = float(size) * 8 / (H * W)
    print(f"Compressed size: {size} bytes, BPP: {bpp:.4f}")
    
    # Decompress
    print("Decompressing image...")
    with Path(output_file).open("rb") as f:
        original_size = read_uints(f, 2)
        strings, shape = read_body(f)
    
    out = net.decompress(strings, shape, text_embeddings)
    x_hat = out["x_hat"].detach().clone()
    x_hat = x_hat[:, :, 0:original_size[0], 0:original_size[1]]
    
    # Calculate metrics
    psnr = compute_psnr(x, x_hat)
    try:
        ms_ssim = ms_ssim_func(x, x_hat, data_range=1.).item()
    except:
        ms_ssim = ms_ssim_func(torchvision.transforms.Resize(256)(x), torchvision.transforms.Resize(256)(x_hat), data_range=1.).item()
    
    lpips_score = loss_fn_alex(x, x_hat).item()
    
    # Save reconstructed image
    output_image = os.path.join(args.output_dir, "reconstructed.png")
    torchvision.utils.save_image(x_hat, output_image)
    
    # Save original image
    original_image = os.path.join(args.output_dir, "original.png")
    torchvision.utils.save_image(x, original_image)
    
    # Print results
    print(f"\nResults:")
    print(f"BPP: {bpp:.4f}")
    print(f"PSNR: {psnr:.4f}")
    print(f"MS-SSIM: {ms_ssim:.4f}")
    print(f"LPIPS: {lpips_score:.4f}")
    print(f"\nOriginal image saved to: {original_image}")
    print(f"Reconstructed image saved to: {output_image}")
    print(f"Compressed file saved to: {output_file}")

if __name__ == "__main__":
    main()
