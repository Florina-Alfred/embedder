import argparse
import logging
import os

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim


# ------------------ Backbone Definition ------------------
logger = logging.getLogger(__name__)


class Backbone(nn.Module):
    """Small wrapper around a timm backbone that returns spatial features.

    By default the class does NOT apply a learned projection (visualization mode).
    Set `use_projection=True` to add a 1x1 conv that maps channels to `out_channels`.
    """

    def __init__(
        self,
        model_name: str = "convnext_tiny",
        out_channels: int = 512,
        use_projection: bool = False,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,  # returns spatial feature maps
            out_indices=(3,),  # last stage
        )
        backbone_channels = self.backbone.feature_info.channels()[-1]
        if use_projection:
            self.projection: nn.Module = nn.Conv2d(
                backbone_channels, out_channels, kernel_size=1
            )
        else:
            # For visualization we prefer to keep pretrained features intact
            self.projection = nn.Identity()

        # Freeze backbone parameters by default (projection may remain trainable)
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)[0]  # [B, C, H, W]
        return self.projection(features)


# ------------------ Setup ------------------
def preprocess(frame: np.ndarray, size: tuple[int, int] = (224, 224)) -> torch.Tensor:
    """Resize, convert BGR->RGB, scale and apply ImageNet normalization.

    Returns a batched float32 tensor of shape [1, 3, H, W].
    """
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_norm = frame_rgb.astype(np.float32) / 255.0
    frame_chw = np.transpose(frame_norm, (2, 0, 1))  # HWC -> CHW

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    frame_chw = (frame_chw - mean) / std

    tensor = torch.from_numpy(frame_chw).unsqueeze(0)
    return tensor


def train_projection_online(
    model: Backbone,
    device: torch.device,
    steps: int = 200,
    lr: float = 1e-3,
    save_path: str = "projection_trained.pt",
    camera_index: int = 0,
):
    """Train the model's 1x1 projection (online) with a reconstruction objective.

    This function builds a small decoder (1x1 conv) and trains projection+decoder to
    reconstruct backbone features. The backbone itself remains frozen.
    """
    # Check projection availability
    if not hasattr(model, "projection") or isinstance(model.projection, nn.Identity):
        logger.error(
            "Model projection is Identity or missing; enable --use-projection to train."
        )
        return

    # Prepare model and decoder for training
    # Ensure backbone stays frozen
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Infer channel dims
    try:
        in_ch = model.backbone.feature_info.channels()[-1]
    except Exception:
        # fallback: inspect a forward pass
        logger.debug(
            "feature_info unavailable, running a dummy forward pass to infer channels"
        )
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            out = model.backbone(dummy)[0]
        in_ch = out.shape[1]

    out_ch = getattr(model.projection, "out_channels", None)
    if out_ch is None:
        # fallback if projection is wrapped or custom
        out_ch = in_ch

    decoder = nn.Conv2d(out_ch, in_ch, kernel_size=1).to(device)

    # Put model/projection/decoder into float32 for stable training
    model.float()
    decoder.float()
    model.train()
    model.projection.train()

    params = list(model.projection.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=lr)
    criterion = nn.MSELoss()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Could not open camera for training (index=%d)", camera_index)
        return

    try:
        for step in range(steps):
            # read frame (skip if not returned)
            ret, frame = cap.read()
            if not ret:
                logger.debug("Empty frame while training; retrying")
                continue

            input_tensor = preprocess(frame).to(device).float()

            # Extract backbone features as target (no grads)
            with torch.no_grad():
                target_feats = model.backbone(input_tensor)[0]

            pred = model.projection(target_feats)
            recon = decoder(pred)

            loss = criterion(recon, target_feats)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                logger.info("[train] step %d/%d loss=%.6f", step, steps, float(loss))

    except Exception:
        logger.exception("Exception during projection training")
    finally:
        cap.release()
        # Save projection state
        try:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(model.projection.state_dict(), save_path)
            logger.info("Saved trained projection to %s", save_path)
        except Exception:
            logger.exception("Failed to save projection to %s", save_path)

    # Return to eval mode and (optionally) half precision on CUDA
    model.eval()
    if device.type == "cuda":
        model.half()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Camera embedding visualizer + optional projection trainer"
    )
    parser.add_argument(
        "--use-projection",
        action="store_true",
        help="Enable learned 1x1 projection (trainable)",
    )
    parser.add_argument(
        "--projection-path",
        type=str,
        default=None,
        help="Path to load projection state (pt file)",
    )
    parser.add_argument(
        "--train-projection",
        action="store_true",
        help="Train the 1x1 projection using camera frames",
    )
    parser.add_argument(
        "--projection-out-channels",
        type=int,
        default=512,
        help="Number of output channels for the projection",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=200,
        help="Number of training steps when --train-projection is set",
    )
    parser.add_argument(
        "--train-lr",
        type=float,
        default=1e-3,
        help="Learning rate for projection training",
    )
    parser.add_argument(
        "--save-projection",
        type=str,
        default="projection_trained.pt",
        help="File to save trained projection weights",
    )
    parser.add_argument(
        "--camera-index", type=int, default=0, help="Camera index for cv2.VideoCapture"
    )

    # Preset/model selection
    parser.add_argument(
        "-q",
        "--quality",
        choices=["lite", "mid", "high"],
        default="mid",
        help="Model quality preset: lite->convnext_small, mid->swin_small, high->vit_base",
    )
    parser.add_argument(
        "-s",
        "--semantic",
        choices=["none", "clip", "dino"],
        default="none",
        help="Use semantic model pipelines (clip/dino) instead of preset",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        default=None,
        help="Explicit timm/model name override (wins over quality preset)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Resolve model name from quality preset unless explicit model-name is provided
    preset_map = {
        "lite": "convnext_small",
        "mid": "swin_small_patch4_window7_224",
        "high": "vit_base_patch16_224",
    }
    resolved_model = args.model_name or preset_map.get(
        args.quality, "swin_small_patch4_window7_224"
    )

    if args.semantic != "none":
        logger.info(
            "Semantic mode set to %s; semantic pipelines will be used if implemented",
            args.semantic,
        )

    model = Backbone(
        model_name=resolved_model,
        out_channels=args.projection_out_channels,
        use_projection=args.use_projection,
    ).to(device)
    model.eval()
    if device.type == "cuda":
        model.half()

    # Load projection weights if requested
    if args.projection_path:
        try:
            ckpt = torch.load(args.projection_path, map_location=device)
            try:
                # try loading whole checkpoint into model (partial load allowed)
                model.load_state_dict(ckpt, strict=False)
                logger.info("Loaded checkpoint into model (partial load allowed)")
            except Exception:
                # try loading projection-only state dict
                try:
                    if hasattr(model, "projection") and isinstance(
                        model.projection, nn.Module
                    ):
                        model.projection.load_state_dict(ckpt)
                        logger.info(
                            "Loaded projection state dict from %s", args.projection_path
                        )
                except Exception:
                    logger.exception(
                        "Failed to load projection weights from %s",
                        args.projection_path,
                    )
        except Exception:
            logger.exception(
                "Could not read projection checkpoint: %s", args.projection_path
            )

    # Train projection if requested
    if args.train_projection:
        if not args.use_projection:
            logger.error(
                "--train-projection requires --use-projection. Exiting training step."
            )
        else:
            train_projection_online(
                model=model,
                device=device,
                steps=args.train_steps,
                lr=args.train_lr,
                save_path=args.save_projection,
                camera_index=args.camera_index,
            )

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        logger.error(
            "Could not open camera (cv2.VideoCapture(%d)). Exiting.", args.camera_index
        )
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera returned empty frame — exiting loop.")
                break

            # Preprocess and move to device; keep dtype float32 then cast to half on CUDA
            input_tensor = preprocess(frame).to(device)
            if device.type == "cuda":
                input_tensor = input_tensor.half()

            with torch.no_grad():
                features = model(input_tensor)  # could be tensor or list/tuple

            # Ensure tensor shape
            if isinstance(features, (list, tuple)):
                feats = features[0]
            else:
                feats = features

            # If model returned token embeddings (e.g. ViT: [B, N+1, D]), reshape to [B, D, Hf, Wf]
            if feats.dim() == 3:
                b, n, d = feats.shape
                # Heuristic: tokens format [B, N+1, D] where N is num patches
                if n > d:
                    # assume tokens, drop cls token if present
                    patches = feats[:, 1:, :] if n > 1 else feats
                    N = patches.shape[1]
                    h = int(round(N**0.5))
                    if h * h == N:
                        feats = patches.transpose(1, 2).reshape(b, d, h, h)
                        logger.debug("Reshaped ViT tokens to spatial %s", feats.shape)
                    else:
                        # fallback: treat channel dim as d and collapse to single map
                        feats = feats.transpose(1, 2).reshape(b, d, 1, N)
                        logger.debug("Fallback reshape of tokens to %s", feats.shape)

            # Convert to float for stable CPU ops
            feats = feats.float()

            # Aggregate activation energy across channels (sum(abs)) for a stable visualization
            energy = feats[0].abs().sum(dim=0).cpu().numpy()

            # Percentile-based clipping to reduce outlier effects
            vmin = float(np.percentile(energy, 1.0))
            vmax = float(np.percentile(energy, 99.0))
            if vmax - vmin < 1e-6:
                # fallback to global running max to avoid flat maps
                if not hasattr(main, "_energy_ema"):
                    main._energy_ema = vmax
                main._energy_ema = max(main._energy_ema * 0.99, vmax)
                scale = main._energy_ema + 1e-6
                energy = energy / scale
            else:
                energy = np.clip(energy, vmin, vmax)
                energy = (energy - vmin) / (vmax - vmin + 1e-6)

            H_orig, W_orig = frame.shape[:2]
            energy_uint8 = (energy * 255).astype(np.uint8)
            # Upsample to original frame size and apply light smoothing
            energy_up = cv2.resize(
                energy_uint8, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR
            )
            # apply small bilateral filter to preserve edges while smoothing
            try:
                energy_up = cv2.bilateralFilter(
                    energy_up, d=9, sigmaColor=75, sigmaSpace=75
                )
            except Exception:
                energy_up = cv2.GaussianBlur(energy_up, (7, 7), 0)

            heatmap = cv2.applyColorMap(energy_up, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

            cv2.imshow("Camera Feed", frame)
            cv2.imshow("Embedding Overlay", overlay)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
