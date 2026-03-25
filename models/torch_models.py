"""Extract reference outputs from HuggingFace models for use in tests.

Run once to generate tests/data/dinov3_vits16_reference.npz and
tests/data/siglip2_vitb16_256_reference.npz:

    uv run python torch_models.py
"""

from pathlib import Path

import numpy as np
import torch
from transformers import pipeline

OUTPUT_DIR = Path(__file__).parent / "tests" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
img_np = rng.standard_normal((3, 256, 256)).astype(np.float32)
x = torch.from_numpy(img_np).unsqueeze(0)  # (1, 3, 256, 256)

# --- DINOv3 ViT-S/16 (LVD-1689M) ---
# Output layout: [cls, reg0, reg1, reg2, reg3, patch0, ..., patch255]
feature_extractor = pipeline(
    model="facebook/dinov3-vits16-pretrain-lvd1689m",
    task="image-feature-extraction",
)
model = feature_extractor.model.eval()

with torch.no_grad():
    out = model(x).last_hidden_state  # (1, 261, 384)

cls_token = out[0, 0].numpy()  # (384,)

output = OUTPUT_DIR / "dinov3_vits16_reference.npz"
np.savez(output, cls_token=cls_token, img=img_np)
print(f"Saved DINOv3 reference to {output}")
print(f"  cls_token shape: {cls_token.shape}")
print(f"  cls_token[:4]:   {cls_token[:4]}")

# --- SigLIP2 ViT-B/16 at 256×256 ---
# No cls or reg tokens; all 256 positions are patch tokens.
feature_extractor = pipeline(
    model="google/siglip2-base-patch16-256",
    task="image-feature-extraction",
)
model = feature_extractor.model.vision_model.eval()

with torch.no_grad():
    out = model(x).last_hidden_state  # (1, 256, 768)

patch_tokens = out[0].numpy()  # (256, 768)

output = OUTPUT_DIR / "siglip2_vitb16_256_reference.npz"
np.savez(output, patch_tokens=patch_tokens, img=img_np)
print(f"Saved SigLIP2 reference to {output}")
print(f"  patch_tokens shape: {patch_tokens.shape}")
print(f"  patch_tokens[0, :4]: {patch_tokens[0, :4]}")

# --- EUPE ViT-T/16 ---
import sys

DIR = Path("~/.cache/torch/hub/eupe").expanduser()
if str(DIR) not in sys.path:
    sys.path.insert(0, str(DIR))

torch_hub_cfg = {
    "repo_or_dir": str(DIR / "EUPE"),
    "model": "eupe_vitt16",
    "source": "local",
    "pretrained": True,
    "weights": str((Path("~/.cache/torch/hub/eupe/weights/EUPE-ViT-T.pt")).expanduser()),
}

try:
    model = torch.hub.load(**torch_hub_cfg).eval()
    
    # 224x224 is the standard size for EUPE ViT-T
    img_np_224 = rng.standard_normal((3, 224, 224)).astype(np.float32)
    x_224 = torch.from_numpy(img_np_224).unsqueeze(0)
    
    with torch.no_grad():
        out = model.forward_features(x_224)["x_prenorm"]  # Contains cls/reg + patch tokens
        
    features = out.numpy()
    
    output = OUTPUT_DIR / "eupe_vitt16_reference.npz"
    np.savez(output, features=features, img=img_np_224)
    print(f"Saved EUPE reference to {output}")
    print(f"  features shape: {features.shape}")
except Exception as e:
    print(f"Failed to generate EUPE reference: {e}")
