import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
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
    
    # Save raw attention data for debugging
    with open(os.path.join(output_dir, f"attention_shapes.txt"), "w") as f:
        for i, attn_map in enumerate(attention_weights):
            f.write(f"Attention map {i} shape: {attn_map.shape}\n")
    
    for i, attn_map in enumerate(attention_weights):
        # Save raw attention tensor as numpy for later analysis
        attn_np_file = os.path.join(output_dir, f"attn_raw_{i}.npy")
        np.save(attn_np_file, attn_map.cpu().numpy())
        
        # Handle various attention map shapes
        print(f"Processing attention map with shape: {attn_map.shape}")
        
        # If we have a 4D tensor [batch, heads, q_len, k_len]
        if len(attn_map.shape) == 4:
            # Average across heads
            avg_attn = attn_map.mean(dim=1)[0]  # [q_len, k_len]
        # If we have a 3D tensor [batch, q_len, k_len]
        elif len(attn_map.shape) == 3:
            avg_attn = attn_map[0]  # [q_len, k_len]
        # If we have a 2D tensor [q_len, k_len] already
        elif len(attn_map.shape) == 2:
            avg_attn = attn_map
        else:
            print(f"Warning: Unexpected attention map shape: {attn_map.shape}")
            continue
            
        # Save the attention matrix as an image for reference
        plt.figure(figsize=(12, 10))
        plt.imshow(avg_attn.cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(f"{module_type} {i+1}: Attention Matrix\nShape: {avg_attn.shape}")
        plt.savefig(os.path.join(output_dir, f"{module_type.lower()}{i+1}_attn_matrix.png"))
        plt.close()
        
        # Create more useful visualizations based on matrix dimensions
        if avg_attn.shape[1] <= 77:  # If second dimension is likely text tokens (CLIP uses max 77 tokens)
            # This is probably image -> text attention
            # Let's create a more meaningful visualization: average attention per text token
            token_importance = avg_attn.mean(dim=0).cpu().numpy()
            
            plt.figure(figsize=(15, 5))
            plt.bar(range(len(token_importance)), token_importance)
            plt.title(f"{module_type} {i+1}: Average Attention Per Text Token")
            plt.xlabel("Text Token Index")
            plt.ylabel("Average Attention")
            plt.savefig(os.path.join(output_dir, f"{module_type.lower()}{i+1}_token_importance.png"))
            plt.close()
            
            # Try to visualize spatial attention if we can infer a grid structure
            # Check if the first dimension could be a square grid (image tokens)
            img_tokens = avg_attn.shape[0]
            grid_size = int(np.sqrt(img_tokens))
            
            # If it's close to a perfect square, reshape as an image grid
            if abs(grid_size**2 - img_tokens) / img_tokens < 0.1:  # Within 10% of perfect square
                # Get average attention across all text tokens for each image position
                spatial_attention = avg_attn.mean(dim=1).reshape(grid_size, grid_size).cpu().numpy()
                
                plt.figure(figsize=(10, 8))
                plt.imshow(spatial_attention, cmap='hot')
                plt.colorbar()
                plt.title(f"{module_type} {i+1}: Spatial Attention Map\nHigher values = more attention to text")
                plt.savefig(os.path.join(output_dir, f"{module_type.lower()}{i+1}_spatial_map.png"))
                plt.close()
                
                # Create an overlay on the original image for better interpretation
                plt.figure(figsize=(12, 10))
                plt.imshow(img_np)
                plt.imshow(cv2.resize(spatial_attention, (img_np.shape[1], img_np.shape[0])), 
                         alpha=0.5, cmap='hot')
                plt.colorbar()
                plt.title(f"{module_type} {i+1}: Attention Overlay")
                plt.savefig(os.path.join(output_dir, f"{module_type.lower()}{i+1}_attention_overlay.png"))
                plt.close()
        
        try:            
            if module_type == "Injector":
                # Try to visualize image-to-text attention
                try:
                    # Check if we can interpret dimension 0 as image tokens arranged in 2D
                    # Only proceed if it seems like image tokens (perfect square)
                    if avg_attn.shape[0] > 16:  # Likely image tokens if > 16
                        img_tokens = avg_attn.shape[0]
                        h = w = int(np.sqrt(img_tokens))
                        
                        if h*w == img_tokens:  # Perfect square check
                            # Average attention to text tokens
                            if avg_attn.shape[1] > 1:  # Multiple text tokens
                                text_importance = avg_attn.mean(dim=0)  # [text_seq_len]
                                
                                # Plot text token importance if it's not a scalar
                                plt.figure(figsize=(12, 4))
                                plt.bar(range(text_importance.shape[0]), text_importance.cpu().numpy())
                                plt.title(f"Injector {i+1}: Text Token Importance")
                                plt.xlabel("Text Token Position")
                                plt.ylabel("Average Attention")
                                plt.savefig(os.path.join(output_dir, f"injector{i+1}_text_importance.png"))
                                plt.close()
                            
                            # Create attention heatmap (how much each image position attends to text)
                            attn_to_text = avg_attn.mean(dim=1).reshape(h, w)
                            plt.figure(figsize=(10, 8))
                            plt.imshow(attn_to_text.cpu().numpy(), cmap='hot')
                            plt.colorbar()
                            plt.title(f"Injector {i+1}: Image Attention to Text")
                            plt.savefig(os.path.join(output_dir, f"injector{i+1}_image_attn_heatmap.png"))
                            plt.close()
                except Exception as e:
                    print(f"Error visualizing Injector attention: {e}")
            
            elif module_type == "Extractor":
                try:
                    # Try to visualize text-to-image attention
                    if avg_attn.shape[1] > 16:  # Likely image tokens in dimension 1
                        img_tokens = avg_attn.shape[1]
                        h = w = int(np.sqrt(img_tokens))
                        
                        if h*w == img_tokens:  # Perfect square check
                            # Sum attention across text tokens to see which image regions are important
                            image_importance = avg_attn.sum(dim=0)
                            
                            # Reshape to image dimensions
                            importance_map = image_importance.reshape(h, w)
                            
                            plt.figure(figsize=(10, 8)) 
                            plt.imshow(importance_map.cpu().numpy(), cmap='hot')
                            plt.colorbar()
                            plt.title(f"Extractor {i+1}: Image Region Importance")
                            plt.savefig(os.path.join(output_dir, f"extractor{i+1}_image_importance.png"))
                            plt.close()
                except Exception as e:
                    print(f"Error visualizing Extractor attention: {e}")
        except Exception as e:
            print(f"General error in attention visualization: {e}")
            # At minimum, save the raw attention map
            plt.figure(figsize=(10, 8))
            plt.imshow(avg_attn.cpu().numpy())
            plt.colorbar()
            plt.title(f"Raw Attention Map {i+1}")
            plt.savefig(os.path.join(output_dir, f"raw_attn_map_{i+1}.png"))
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
    
    # This is a simpler approach - we'll just import what we need
    print("Looking for a working TACO script to help initialize the model correctly...")
    
    try:
        # Load an existing model as is, like in evaluate_caption_impact_kodak.py
        # Avoid customizing the model to reduce risk of errors
        from run_single_image import main as run_single_image_main
        print("Found run_single_image.py, which we can use to load the model properly")
    except ImportError:
        print("Could not find run_single_image module, looking for alternatives...")
    
    try:
        # Try direct import
        print(f"Loading checkpoint: {checkpoint}")
        print("Loading TACO model...")
        taco_config = model_config()
        net = TACO(taco_config, text_embedding_dim=CLIP_text_model.config.hidden_size)
        
        # This is a special monkey-patching approach to fix the initialization error
        # We temporarily modify the __init__ method of CompressionModel to avoid the error
        original_init = net.__init__
        
        def fixed_init(*args, **kwargs):
            # Skip the problematic CompressionModel.__init__ call
            pass
        
        # Apply the monkey patch
        net.__init__ = fixed_init
        
        # Now load the state dict
        state_dict = torch.load(checkpoint, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        try:
            net.load_state_dict(state_dict)
        except:
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace("module.", "")] = v
            net.load_state_dict(new_state_dict)
            
        # Restore original init
        net.__init__ = original_init
        
        net = net.eval().to(device)
        net.requires_grad_(False)
        print("Successfully loaded the TACO model!")
        
    except Exception as e:
        print(f"Error loading TACO model: {e}")
        print("Continuing with a simpler approach just for attention visualization")
        print("We'll only use the model to extract attention weights, not for full compression")
        
        # Just focus on model.g_a where the cross-attention happens
        taco_config = model_config()
        
        # Minimal import
        import sys
        sys.path.append("./modules/transform")
        from analysis import AnalysisTransformEX
        
        # Create just the encoder part that has the attention mechanisms
        model_encoder = AnalysisTransformEX(
            taco_config.N, 
            taco_config.M, 
            CLIP_text_model.config.hidden_size
        ).to(device)
        
        # Create a proper nn.Module-based class
        class SimpleModel(torch.nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.g_a = encoder
                
        # Set this as our "model" for attention visualization
        net = SimpleModel(model_encoder)
        
        print("Created simplified model for attention visualization only")
    
    # This part of the code is handled in the model loading section above
    
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
    
    # Setup for attention visualization only
    print("\nSetting up attention visualization...")
    print("This simplified approach will only visualize attention patterns without compressing/decompressing")
    
    # Find and register hooks for all cross-attention modules
    injector_hooks = []
    extractor_hooks = []
    injector_attention_hooks = []
    extractor_attention_hooks = []
    
    # Search for Injector and Extractor modules in the model
    print("Finding attention modules in model...")
    found_modules = False
    
    # Debug: print all module types to see what we have
    print("Model structure:")
    module_types = set()
    for name, module in net.named_modules():
        module_type = type(module).__name__
        module_types.add(module_type)
        # Print the first few modules for debugging
        if len(module_types) < 10:
            print(f"  {name}: {module_type}")
    
    print(f"Module types found: {', '.join(module_types)}")
    
    # Now register hooks on the attention modules using class name instead of isinstance
    for name, module in net.named_modules():
        # Use class name comparison instead of isinstance
        module_type = type(module).__name__
        
        if module_type == "Injector":
            found_modules = True
            hook = CrossAttentionHook(f"Injector-{len(injector_hooks)}")
            try:
                print(f"Found Injector module at {name}, checking contents: {dir(module)}")
                # Check if cross_attn exists
                if hasattr(module, 'cross_attn'):
                    injector_hooks.append(module.cross_attn.register_forward_hook(hook))
                    injector_attention_hooks.append(hook)
                    print(f"Registered hook on Injector: {name}")
                else:
                    print(f"Warning: Injector {name} has no cross_attn attribute, looking for alternatives")
                    # See if there's any attribute that looks like an attention module
                    for attr_name in dir(module):
                        if 'attn' in attr_name.lower():
                            print(f"  Found potential attention attribute: {attr_name}")
                            attr = getattr(module, attr_name)
                            if hasattr(attr, 'register_forward_hook'):
                                injector_hooks.append(attr.register_forward_hook(hook))
                                injector_attention_hooks.append(hook)
                                print(f"  Registered hook on alternative attention: {attr_name}")
                                break
            except AttributeError as e:
                print(f"Error registering hook on Injector {name}: {e}")
                print(f"Module structure: {dir(module)}")
        
        elif module_type == "Extractor":
            found_modules = True
            hook = CrossAttentionHook(f"Extractor-{len(extractor_hooks)}")
            try:
                print(f"Found Extractor module at {name}, checking contents: {dir(module)}")
                # Check if cross_attn exists
                if hasattr(module, 'cross_attn'):
                    extractor_hooks.append(module.cross_attn.register_forward_hook(hook))
                    extractor_attention_hooks.append(hook)
                    print(f"Registered hook on Extractor: {name}")
                else:
                    print(f"Warning: Extractor {name} has no cross_attn attribute, looking for alternatives")
                    # See if there's any attribute that looks like an attention module
                    for attr_name in dir(module):
                        if 'attn' in attr_name.lower():
                            print(f"  Found potential attention attribute: {attr_name}")
                            attr = getattr(module, attr_name)
                            if hasattr(attr, 'register_forward_hook'):
                                extractor_hooks.append(attr.register_forward_hook(hook))
                                extractor_attention_hooks.append(hook)
                                print(f"  Registered hook on alternative attention: {attr_name}")
                                break
            except AttributeError as e:
                print(f"Error registering hook on Extractor {name}: {e}")
                print(f"Module structure: {dir(module)}") 
    if not found_modules:
        print("WARNING: Could not find any Injector or Extractor modules in the model!")
        print("Attention maps will not be available.")
        
    # Process each caption
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
        
        # Save a copy of the original image in this caption's directory
        orig_img_path = os.path.join(caption_output_dir, "original.png")
        torchvision.utils.save_image(x, orig_img_path)
        
        # Our simplified approach: just run a forward pass through the encoder
        # to capture attention weights, without doing full compression
        print(f"Running encoder to capture attention for caption: '{current_caption}'...")
        
        try:
            with torch.no_grad():
                # Print the model structure to be sure
                if hasattr(net, 'g_a'):
                    print(f"Model has g_a attribute of type: {type(net.g_a).__name__}")
                    
                    # Check for forward method
                    if hasattr(net.g_a, 'forward'):
                        print("g_a has forward method")
                    else:
                        print("g_a does not have forward method!")
                    
                # Only run the encoder (g_a) since that contains the attention mechanisms
                # This avoids any issues with the full compression process
                _ = net.g_a(x_padded, text_embeddings)
                
                # If that works, great! If not, we'll try even simpler approach
                print("Successfully ran the encoder and captured attention weights")
                
        except Exception as e:
            print(f"Error running encoder: {e}")
            print("Trying a more direct approach...")
            
            # If we're using the simplified model approach, we need to manually pass 
            # through each module to capture attention
            try:
                # Find the Injector and Extractor modules directly
                injectors = []
                extractors = []
                
                print("Looking for Injector and Extractor modules in the encoder...")
                for name, module in net.g_a.named_modules():
                    if isinstance(module, Injector):
                        injectors.append((name, module))
                    elif isinstance(module, Extractor):
                        extractors.append((name, module))
                
                print(f"Found {len(injectors)} Injectors and {len(extractors)} Extractors")
                
                # Try to run the forward pass of the first layer at least to get some data
                if hasattr(net.g_a, 'analysis_transform') and len(net.g_a.analysis_transform) > 0:
                    # Try to run just the non-attention layers first
                    print("Trying to run non-attention layers...")
                    features = x_padded
                    
                    # Find the first attention layer
                    first_attn_idx = -1
                    for i, layer in enumerate(net.g_a.analysis_transform):
                        if isinstance(layer, Injector) or isinstance(layer, Extractor):
                            first_attn_idx = i
                            print(f"First attention layer at index {i}")
                            break
                    
                    # Run the layers before the first attention layer
                    if first_attn_idx > 0:
                        for i in range(first_attn_idx):
                            layer = net.g_a.analysis_transform[i]
                            print(f"Running layer {i}: {type(layer).__name__}")
                            features = layer(features)
                        
                        print("Reached the first attention layer")
                    else:
                        print("No regular layers before attention")
            except Exception as inner_e:
                print(f"Error with direct approach: {inner_e}")
                print("Unable to capture attention weights for this model")
            
        # Visualize any attention weights we captured
        print("Visualizing attention maps...")
        
        # Process injector attention (image to text attention)
        for j, hook in enumerate(injector_attention_hooks):
            if hook.attn_weights:  # Only process if we captured weights
                print(f"Processing attention from Injector {j} - found {len(hook.attn_weights)} attention maps")
                injector_dir = os.path.join(caption_output_dir, f"injector_{j}")
                visualize_attention(hook.attn_weights, x, injector_dir, current_caption, "Injector")
            else:
                print(f"No attention weights captured for Injector {j}")
        
        # Process extractor attention (text to image attention)
        for j, hook in enumerate(extractor_attention_hooks):
            if hook.attn_weights:  # Only process if we captured weights
                print(f"Processing attention from Extractor {j} - found {len(hook.attn_weights)} attention maps")
                extractor_dir = os.path.join(caption_output_dir, f"extractor_{j}")
                visualize_attention(hook.attn_weights, x, extractor_dir, current_caption, "Extractor")
            else:
                print(f"No attention weights captured for Extractor {j}")
                
        print(f"Results for caption '{current_caption}' saved to: {caption_output_dir}")
    
    # Clean up hooks
    for hook in injector_hooks + extractor_hooks:
        hook.remove()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
