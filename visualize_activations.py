import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

from models import TACO
from config.config import model_config

from transformers import CLIPTextModel, AutoTokenizer

# --- 1. Setup device ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. Load model and checkpoint ---
clip_model_name = "openai/clip-vit-base-patch32"
CLIP_text_model = CLIPTextModel.from_pretrained(clip_model_name).to(device)
CLIP_tokenizer = AutoTokenizer.from_pretrained(clip_model_name)

taco_config = model_config()
net = TACO(taco_config, text_embedding_dim=CLIP_text_model.config.hidden_size)
net = net.eval().to(device)

# Find a checkpoint in the checkpoint folder
ckpt_files = [f for f in os.listdir('checkpoint') if f.endswith('.pth.tar')]
assert ckpt_files, "No checkpoint found in checkpoint/"
ckpt_path = os.path.join('checkpoint', ckpt_files[0])
print(f"Loading checkpoint: {ckpt_path}")
state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
try:
    net.load_state_dict(state_dict)
except:
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    net.load_state_dict(new_state_dict)
net.requires_grad_(False)
net.update()

# --- 3. Prepare input image from Kodak ---
kodak_files = [f for f in os.listdir('kodak') if f.endswith('.png') or f.endswith('.jpg')]
assert kodak_files, "No images found in kodak/"
img_path = os.path.join('kodak', kodak_files[0])
print(f"Using image: {img_path}")
img = Image.open(img_path).convert('RGB')
transform = T.Compose([T.ToTensor()])
input_image = transform(img).unsqueeze(0).to(device)

# --- 4. Prepare dummy caption and get text embedding ---
caption = ""  # Try also with "random text" or a real caption
clip_token = CLIP_tokenizer([caption], padding="max_length", max_length=38, truncation=True, return_tensors="pt").to(device)
text_embeddings = CLIP_text_model(**clip_token).last_hidden_state

# --- 5. Register hooks for activations ---
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu()
    return hook

layer_indices = {
    'before_injector1': 7,
    'after_injector1': 8,
    'after_attention1': 9,
    'before_extractor': 13,
    'after_extractor': 14,
    'after_injector2': 15
}
handles = []
for name, idx in layer_indices.items():
    handles.append(net.g_a.analysis_transform[idx].register_forward_hook(get_activation(name)))

# --- 6. Run forward pass ---
with torch.no_grad():
    _ = net(input_image, text_embeddings)

# --- 7. Visualize activations ---
for name in layer_indices.keys():
    act = activations[name][0]  # First image in batch
    print(f"{name} activation shape: {act.shape}")
    if act.ndim == 3:  # (C, H, W)
        fig, axes = plt.subplots(1, min(8, act.shape[0]), figsize=(20, 3))
        for i in range(min(8, act.shape[0])):
            if act[i].ndim == 2:
                axes[i].imshow(act[i].numpy(), cmap='viridis')
                axes[i].set_title(f'{name} - ch {i}')
                axes[i].axis('off')
            else:
                axes[i].set_title(f'{name} - ch {i} (not 2D)')
                axes[i].axis('off')
        plt.suptitle(f'Activations at {name}')
        plt.show()
    else:
        print(f"Skipping {name}: activation is not 3D (got shape {act.shape})")

# --- 8. Remove hooks ---
for handle in handles:
    handle.remove()