import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math
import lpips
import torch.nn.functional as F
from pytorch_msssim import ms_ssim as ms_ssim_func
from transformers import CLIPTextModel, AutoTokenizer

from config.config import model_config 
from models import TACO
from utils.utils import *
from modules.transform.analysis import Injector, Extractor
from modules.transform.context import ChannelContextEX
from modules.transform.entropy import EntropyParametersEX

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

# Custom hook for capturing cross-attention weights from Injector/Extractor
class CrossAttentionHook:
    def __init__(self, name):
        self.name = name
        self.attn_weights = []
        self.count = 0

    def __call__(self, module, input_args, output):
        # MultiheadAttention in PyTorch returns output and attention weights
        # We only care about the attention weights (second element)
        if isinstance(output, tuple) and len(output) > 1:
            # Extract attention weights
            attn_weights = output[1].detach().cpu()
            self.attn_weights.append(attn_weights)
            self.count += 1

def visualize_attention(attention_weights, image, output_dir, caption, module_type):
    """Visualize cross-attention maps between image and text"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original image for reference
    plt.figure(figsize=(10, 8))
    img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    plt.imshow(img_np)
    plt.title(f"Original Image")
    plt.savefig(os.path.join(output_dir, f"original_image.png"))
    plt.close()
    
    for i, attn_map in enumerate(attention_weights):
        # Shape: [batch_size, num_heads, seq_len, seq_len]
        # Average across heads
        avg_attn = attn_map.mean(dim=1)[0]  # [seq_len, seq_len]
        
        if module_type == "Injector":
            # In Injector: image (query) attends to text (key/value)
            # Reshape image tokens to 2D
            h = w = int(np.sqrt(avg_attn.shape[0]))
            
            # Average attention paid to each text token by each image position
            text_importance = avg_attn.mean(dim=0)  # [text_seq_len]
            
            # Plot text token importance
            plt.figure(figsize=(12, 4))
            plt.bar(range(len(text_importance)), text_importance.numpy())
            plt.title(f"Injector {i+1}: Text Token Importance\nCaption: {caption[:30]}...")
            plt.xlabel("Text Token Position")
            plt.ylabel("Average Attention")
            plt.savefig(os.path.join(output_dir, f"injector{i+1}_text_importance.png"))
            plt.close()
            
            # Create attention heatmap (how much each image position attends to text)
            attn_to_text = avg_attn.mean(dim=1).reshape(h, w)
            plt.figure(figsize=(10, 8))
            plt.imshow(attn_to_text, cmap='hot')
            plt.colorbar()
            plt.title(f"Injector {i+1}: Image Attention to Text\nCaption: {caption[:30]}...")
            plt.savefig(os.path.join(output_dir, f"injector{i+1}_image_attn_heatmap.png"))
            plt.close()
            
        elif module_type == "Extractor":
            # In Extractor: text (query) attends to image (key/value)
            # Sum attention across text tokens to see which image regions are important
            image_importance = avg_attn.sum(dim=0)
            
            # Reshape to image dimensions
            h = w = int(np.sqrt(image_importance.shape[0]))
            importance_map = image_importance.reshape(h, w)
            
            plt.figure(figsize=(10, 8)) 
            plt.imshow(importance_map, cmap='hot')
            plt.colorbar()
            plt.title(f"Extractor {i+1}: Image Region Importance\nCaption: {caption[:30]}...")
            plt.savefig(os.path.join(output_dir, f"extractor{i+1}_image_importance.png"))
            plt.close()

def main():
    # Configuration
    image_path = "./kodak/kodim04.png"  # Hardcoded Kodak image path
    caption = "a woman wearing a red hat and a red dress"
    checkpoint = "checkpoint/lambda_0.0004.pth.tar"  # Path to the model checkpoint
    output_dir = "./attention_analysis"  # Directory to save output
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CLIP text model
    clip_model_name = "openai/clip-vit-base-patch32"
    CLIP_text_model = CLIPTextModel.from_pretrained(clip_model_name).to(device)
    CLIP_text_model.requires_grad_(False)
    CLIP_tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
    
    # Load TACO model
    taco_config = model_config()
    
    # Create a subclass that fixes the initialization issue
    class TACO_Fixed(TACO):
        def __init__(self, config, text_embedding_dim, **kwargs):
            # Initialize CompressionModel with the required parameter
            CompressionModel.__init__(self, entropy_bottleneck_channels=config.N)
            
            # Continue with normal TACO initialization but skip the super().__init__() call
            N = config.N
            M = config.M
            
            self.N = N
            self.M = M
            self.text_embedding_dim = text_embedding_dim
            
            slice_num = config.slice_num
            slice_ch = config.slice_ch
            self.quant = config.quant
            self.slice_num = slice_num
            self.slice_ch = slice_ch
            self.g_a = AnalysisTransformEX(N, M, text_embedding_dim, act=nn.ReLU)
            self.g_s = SynthesisTransformEX(N, M, act=nn.ReLU)
            self.h_a = HyperAnalysisEX(N, M, act=nn.ReLU)
            self.h_s = HyperSynthesisEX(N, M, act=nn.ReLU)
            
            # Initialize other components from taco.py
            # Local context model
            self.local_context = nn.ModuleList(
                nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2)
                for i in range(len(slice_ch))
            )
            
            # Channel context model
            self.channel_context = nn.ModuleList(
                ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else None
                for i in range(slice_num)
            )
            
            # Entropy parameters
            self.entropy_parameters_anchor = nn.ModuleList(
                EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
                if i else EntropyParametersEX(in_dim=M * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
                for i in range(slice_num)
            )
            
            self.entropy_parameters_nonanchor = nn.ModuleList(
                EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
                if i else EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
                for i in range(slice_num)
            )
            
            # Gaussian conditional and entropy bottleneck
            self.gaussian_conditional = GaussianConditional(None)
            self.entropy_bottleneck = EntropyBottleneck(config.N)
    
    # Create the fixed model
    net = TACO_Fixed(taco_config, text_embedding_dim=CLIP_text_model.config.hidden_size)
    net = net.eval().to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint}")
    state_dict = torch.load(checkpoint, map_location=device)['state_dict']
    
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
    print(f"Processing image: {image_path}")
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
    
    # Save original image
    original_image = os.path.join(output_dir, "original.png")
    torchvision.utils.save_image(x, original_image)
    
    # Process captions
    captions = [
        caption,  # Main caption
        "A scenic landscape with water",  # Generic caption
        ""  # Empty caption
    ]
    
    # Find and register hooks for all cross-attention modules
    injector_hooks = []
    extractor_hooks = []
    injector_attention_hooks = []
    extractor_attention_hooks = []
    
    # Find Injector and Extractor modules in the model
    for name, module in net.named_modules():
        # Look specifically for MultiheadAttention modules inside Injectors/Extractors
        if isinstance(module, Injector):
            hook = CrossAttentionHook(f"Injector-{len(injector_hooks)}")
            injector_hooks.append(module.cross_attn.register_forward_hook(hook))
            injector_attention_hooks.append(hook)
            print(f"Registered hook on Injector: {name}")
        
        elif isinstance(module, Extractor):
            hook = CrossAttentionHook(f"Extractor-{len(extractor_hooks)}")
            extractor_hooks.append(module.cross_attn.register_forward_hook(hook))
            extractor_attention_hooks.append(hook)
            print(f"Registered hook on Extractor: {name}")
    
    for i, current_caption in enumerate(captions):
        caption_output_dir = os.path.join(output_dir, f"caption_{i}")
        os.makedirs(caption_output_dir, exist_ok=True)
        
        print(f"\nTesting with caption: '{current_caption}'")
        
        # Reset attention hooks for this caption
        for hook in injector_attention_hooks + extractor_attention_hooks:
            hook.attn_weights = []
            hook.count = 0
        
        # Process caption
        if current_caption:
            clip_token = CLIP_tokenizer([current_caption], padding="max_length", max_length=38, truncation=True, return_tensors="pt").to(device)
            text_embeddings = CLIP_text_model(**clip_token).last_hidden_state
        else:
            # For empty caption, create zero embeddings
            text_embeddings = torch.zeros((1, 38, CLIP_text_model.config.hidden_size)).to(device)
        
        # Compress
        print("Compressing image...")
        out_enc = net.compress(x_padded, text_embeddings)
        shape = out_enc["shape"]
        
        # Visualize the collected attention weights
        print("Visualizing attention maps...")
        # Process injector attention (image to text attention)
        for j, hook in enumerate(injector_attention_hooks):
            if hook.attn_weights:  # Only process if we captured weights
                print(f"Processing attention from {hook.name} - found {len(hook.attn_weights)} attention maps")
                injector_dir = os.path.join(caption_output_dir, f"injector_{j}")
                visualize_attention(hook.attn_weights, x, injector_dir, current_caption, "Injector")
        
        # Process extractor attention (text to image attention)
        for j, hook in enumerate(extractor_attention_hooks):
            if hook.attn_weights:  # Only process if we captured weights
                print(f"Processing attention from {hook.name} - found {len(hook.attn_weights)} attention maps")
                extractor_dir = os.path.join(caption_output_dir, f"extractor_{j}")
                visualize_attention(hook.attn_weights, x, extractor_dir, current_caption, "Extractor")
        
        # Save compressed file and calculate metrics
        output_file = os.path.join(caption_output_dir, "compressed.bin")
        with Path(output_file).open("wb") as f:
            write_uints(f, (H, W))
            write_body(f, shape, out_enc["strings"])
        
        # Calculate BPP
        size = filesize(output_file)
        bpp = float(size) * 8 / (H * W)
        
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
        output_image = os.path.join(caption_output_dir, "reconstructed.png")
        torchvision.utils.save_image(x_hat, output_image)
        
        # Print results
        print(f"\nResults for caption: '{current_caption}'")
        print(f"BPP: {bpp:.4f}")
        print(f"PSNR: {psnr:.4f}")
        print(f"MS-SSIM: {ms_ssim:.4f}")
        print(f"LPIPS: {lpips_score:.4f}")
        print(f"Results saved to: {caption_output_dir}")
    
    # Clean up hooks
    for hook in injector_hooks + extractor_hooks:
        hook.remove()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
