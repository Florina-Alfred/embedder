import cv2
import torch
import torch.nn as nn
import timm
import numpy as np
import os
import torch.nn.functional as F

# ------------------ Backbone Definition ------------------
class Backbone(nn.Module):
    """
    Frozen backbone with double embeddings (512 channels).
    Input:  [B, 3, H, W]
    Output: [B, 512, H_out, W_out]
    """
    def __init__(self, model_name="convnext_tiny", out_channels=512):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,  # gives spatial feature maps
            out_indices=(3,)     # last stage
        )
        backbone_channels = self.backbone.feature_info.channels()[-1]
        self.projection = nn.Conv2d(backbone_channels, out_channels, kernel_size=1)
        
        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)[0]       # [B, C, H, W]
        features = self.projection(features) # [B, out_channels, H, W]
        return features

# ------------------ Setup ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Backbone().to(device).eval()

# Use half precision on GPU if available
if device.type == "cuda":
    model = model.half()

# Create folder to save embeddings
save_dir = "features"
os.makedirs(save_dir, exist_ok=True)

# Initialize camera
cap = cv2.VideoCapture(0)
frame_id = 0

# ------------------ Preprocessing ------------------
def preprocess(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_norm = frame_rgb.astype(np.float32) / 255.0
    frame_chw = np.transpose(frame_norm, (2, 0, 1))  # HWC -> CHW
    tensor = torch.from_numpy(frame_chw).unsqueeze(0)
    return tensor

# ------------------ Main Loop ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame).to(device)
    if device.type == "cuda":
        input_tensor = input_tensor.half()

    with torch.no_grad():
        features = model(input_tensor)  # [1, 512, H_feat, W_feat]

    # ------------------ Overlay Embedding ------------------
    N_channels = 8
    feature_map = features[0, :N_channels, :, :].mean(dim=0, keepdim=True)  # [1, H_feat, W_feat]
    H_orig, W_orig = frame.shape[:2]
    feature_map_up = F.interpolate(feature_map.unsqueeze(0), size=(H_orig, W_orig), mode='bilinear', align_corners=False)
    feature_map_up = feature_map_up[0, 0].cpu().detach().numpy()
    feature_map_up -= feature_map_up.min()
    feature_map_up /= feature_map_up.max()
    feature_map_up = (feature_map_up * 255).astype('uint8')
    heatmap = cv2.applyColorMap(feature_map_up, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    cv2.imshow("Camera Feed", frame)
    cv2.imshow("Embedding Overlay", overlay)

    # ------------------ Optional: Save Embeddings ------------------
    save_path = os.path.join(save_dir, f"frame_{frame_id:05d}.pt")
    torch.save(features.cpu(), save_path)
    frame_id += 1

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
