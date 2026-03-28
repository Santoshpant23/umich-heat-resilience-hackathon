"""
============================================================
AI HEAT RESILIENCE HACKATHON - COMPLETE PIPELINE
============================================================
Phase 1: Run frozen ThermalGen → save baseline predictions
Phase 2: Train refinement net (RGB + ThermalGen + Weather + AlphaEarth)
Phase 3: Generate final test predictions + evaluate

RUN:
  cd ~/AI_Heat_Resilience_Hackathon_Resources/AI_Heat_Resilience_Hackathon_Resources/ThermalGen
  CUDA_VISIBLE_DEVICES=1 python hackathon_pipeline.py --phase 1
  CUDA_VISIBLE_DEVICES=1 python hackathon_pipeline.py --phase 2
  CUDA_VISIBLE_DEVICES=1 python hackathon_pipeline.py --phase 3
============================================================
"""

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from pathlib import Path

# ============================================================
# PATHS — All relative to ThermalGen directory
# ============================================================
BASE = Path(__file__).parent
HACKATHON_ROOT = BASE.parent  # AI_Heat_Resilience_Hackathon_Resources

TRAIN_RGB_DIR = BASE / "datasets_raw" / "Train_2" / "RGB"
TRAIN_THERMAL_DIR = BASE / "datasets_raw" / "Train_2" / "Thermal"
TEST_RGB_DIR = BASE / "datasets_raw" / "Test_2" / "RGB"

SPLIT_JSON = HACKATHON_ROOT / "train_test_split.json"
WEATHER_JSON = HACKATHON_ROOT / "Weather_data_and_code" / "drone_and_weather_metadata.json"
EMBEDDING_DIR = HACKATHON_ROOT / "Google AlphaEarth Embeddings code"

# Output directories
BASELINE_PRED_DIR = BASE / "baseline_predictions"
REFINED_PRED_DIR = BASE / "refined_predictions"
TEST_PRED_DIR = BASE / "test_predictions"
CHECKPOINT_DIR = BASE / "refinement_checkpoints"

# ============================================================
# PHASE 1: Generate ThermalGen baseline predictions
# ============================================================
def phase1_generate_baseline():
    """Run frozen ThermalGen on all train + test RGB images."""
    from thermalgen_demo import ThermalGenSIT

    print("=" * 60)
    print("  PHASE 1: Generating ThermalGen baseline predictions")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Load model
    print("  Loading ThermalGen model...")
    model = ThermalGenSIT.from_pretrained("xjh19972/ThermalGen-L-2-concat").to(device).eval()
    print("  Model loaded!")

    eval_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def run_inference(rgb_dir, output_dir, label=""):
        os.makedirs(output_dir, exist_ok=True)
        files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        print(f"\n  Processing {len(files)} {label} images → {output_dir}")

        for i, fname in enumerate(files):
            out_path = os.path.join(output_dir, fname)
            if os.path.exists(out_path):
                continue  # Skip already generated

            rgb_img = Image.open(os.path.join(rgb_dir, fname)).convert("RGB")
            rgb = eval_transform(rgb_img).unsqueeze(0).to(device)

            with torch.no_grad():
                dataset_idx = torch.ones(1, dtype=torch.long, device=device) * 1000
                pred = model(rgb, dataset_idx)
                pred = pred * 0.5 + 0.5
                pred = torch.clamp(pred, 0, 1)

            pred_img = to_pil_image(pred[0].cpu())
            pred_img.save(out_path)

            if (i + 1) % 20 == 0 or i == 0:
                print(f"    [{i+1}/{len(files)}] {fname}")

        print(f"  Done! {len(files)} predictions saved.")

    # Generate for train and test
    run_inference(TRAIN_RGB_DIR, BASELINE_PRED_DIR / "train", "train")
    run_inference(TEST_RGB_DIR, BASELINE_PRED_DIR / "test", "test")

    print("\n  Phase 1 complete!")
    print(f"  Train predictions: {BASELINE_PRED_DIR / 'train'}")
    print(f"  Test predictions:  {BASELINE_PRED_DIR / 'test'}")


# ============================================================
# DATA LOADING UTILITIES
# ============================================================
def load_metadata():
    """Load train_test_split.json and weather metadata."""
    with open(SPLIT_JSON, 'r') as f:
        split = json.load(f)
    with open(WEATHER_JSON, 'r') as f:
        weather = json.load(f)
    return split, weather


def get_weather_features(thermal_dji_name, weather_data):
    """Extract 8 weather features for a given DJI thermal filename."""
    info = weather_data.get(thermal_dji_name, None)
    if info is None:
        return np.zeros(8, dtype=np.float32)

    features = np.array([
        info.get("temperature_2m", 0.0),
        info.get("relative_humidity_2m", 0.0),
        info.get("total_cloud_cover", 0.0),
        info.get("wind_speed_10m", 0.0),
        info.get("wind_direction_10m", 0.0),
        info.get("direct_radiation", 0.0),
        info.get("diffuse_radiation", 0.0),
        info.get("weather_code", 0.0),
    ], dtype=np.float32)
    return features


def load_alpha_earth_embedding(thermal_dji_name):
    """Load AlphaEarth embedding for a given DJI thermal filename.
    Returns: (64,) vector (spatial average of 24x18x64 grid).
    """
    # Filename pattern: satellite_embedding_DJI_..._T..tif (note double dot)
    emb_name = f"satellite_embedding_{thermal_dji_name}.tif"
    # The actual files have double dot: DJI_..._T..tif
    # thermal_dji_name is like DJI_20240804135946_0314_T.JPG
    # We need: satellite_embedding_DJI_20240804135946_0314_T..tif
    base = thermal_dji_name.replace(".JPG", "")  # DJI_20240804135946_0314_T
    emb_path = EMBEDDING_DIR / f"satellite_embedding_{base}..tif"

    if not emb_path.exists():
        return np.zeros(64, dtype=np.float32)

    try:
        import tifffile
        emb = tifffile.imread(str(emb_path))  # (24, 18, 64)
        # Replace -inf and nan with 0
        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        # Spatial average → (64,)
        emb = emb.mean(axis=(0, 1)).astype(np.float32)
        return emb
    except Exception as e:
        print(f"  Warning: Could not load embedding {emb_path}: {e}")
        return np.zeros(64, dtype=np.float32)


# ============================================================
# REFINEMENT DATASET
# ============================================================
class RefinementDataset(Dataset):
    """
    For each sample, loads:
    - RGB image (3, 256, 256)
    - ThermalGen prediction (1, 256, 256)
    - Ground truth thermal (1, 256, 256)
    - Weather features (8,)
    - AlphaEarth embedding (64,)
    """
    def __init__(self, split_data, weather_data, rgb_dir, thermal_dir, pred_dir, is_test=False):
        self.weather_data = weather_data
        self.rgb_dir = Path(rgb_dir)
        self.thermal_dir = Path(thermal_dir) if thermal_dir else None
        self.pred_dir = Path(pred_dir)
        self.is_test = is_test

        # Build sample list from files that exist in all required dirs
        self.samples = []
        pred_files = set(f for f in os.listdir(pred_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        rgb_files = set(f for f in os.listdir(rgb_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))

        available = pred_files & rgb_files
        if not is_test and thermal_dir:
            thermal_files = set(f for f in os.listdir(thermal_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
            available = available & thermal_files

        for fname in sorted(available):
            # Get DJI thermal name from split
            dji_info = split_data.get(fname, None)
            thermal_dji = dji_info[0] if dji_info else None  # First element is thermal filename
            self.samples.append((fname, thermal_dji))

        self.img_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, thermal_dji = self.samples[idx]

        # Load RGB (3ch)
        rgb = Image.open(self.rgb_dir / fname).convert("RGB")
        rgb = self.img_transform(rgb)  # (3, 256, 256)

        # Load ThermalGen prediction (take first channel)
        pred = Image.open(self.pred_dir / fname).convert("RGB")
        pred = self.img_transform(pred)[:1]  # (1, 256, 256)

        # Weather features (8,)
        weather = get_weather_features(thermal_dji, self.weather_data) if thermal_dji else np.zeros(8, dtype=np.float32)
        weather = torch.from_numpy(weather)

        # AlphaEarth embedding (64,)
        alpha = load_alpha_earth_embedding(thermal_dji) if thermal_dji else np.zeros(64, dtype=np.float32)
        alpha = torch.from_numpy(alpha)

        if not self.is_test:
            # Ground truth thermal
            gt = Image.open(self.thermal_dir / fname).convert("RGB")
            gt = self.img_transform(gt)[:1]  # (1, 256, 256)
            return rgb, pred, weather, alpha, gt, fname
        else:
            return rgb, pred, weather, alpha, fname


# ============================================================
# REFINEMENT NETWORK
# ============================================================
class ConditionInjector(nn.Module):
    """Turns a (B, D) vector into a (B, D, H, W) spatial feature map."""
    def __init__(self, in_dim, out_channels, spatial_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_channels),
        )
        self.spatial_size = spatial_size

    def forward(self, x):
        # x: (B, in_dim)
        x = self.mlp(x)  # (B, out_channels)
        x = x.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        x = x.expand(-1, -1, self.spatial_size, self.spatial_size)  # (B, C, H, W)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class RefinementUNet(nn.Module):
    """
    Lightweight U-Net that refines ThermalGen output.
    Input: RGB(3) + ThermalGen_pred(1) + weather_spatial(4) + alpha_spatial(8) = 16 channels
    Output: refined thermal (1 channel)
    """
    def __init__(self):
        super().__init__()

        # Condition injectors: map weather(8) and alpha(64) to spatial channels
        self.weather_inject = ConditionInjector(8, 4)
        self.alpha_inject = ConditionInjector(64, 8)

        # Input: 3 (RGB) + 1 (ThermalGen pred) + 4 (weather) + 8 (alpha) = 16
        in_ch = 16

        # Encoder
        self.enc1 = ConvBlock(in_ch, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        # Output: residual correction
        self.out_conv = nn.Conv2d(32, 1, 1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, rgb, pred, weather_feat, alpha_feat):
        """
        rgb: (B, 3, 256, 256)
        pred: (B, 1, 256, 256) - ThermalGen prediction
        weather_feat: (B, 8)
        alpha_feat: (B, 64)
        """
        # Inject conditions as spatial feature maps
        w_spatial = self.weather_inject(weather_feat)   # (B, 4, 256, 256)
        a_spatial = self.alpha_inject(alpha_feat)       # (B, 8, 256, 256)

        # Concatenate all inputs
        x = torch.cat([rgb, pred, w_spatial, a_spatial], dim=1)  # (B, 16, 256, 256)

        # Encoder
        e1 = self.enc1(x)       # (B, 32, 256, 256)
        e2 = self.enc2(self.pool(e1))  # (B, 64, 128, 128)
        e3 = self.enc3(self.pool(e2))  # (B, 128, 64, 64)

        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # (B, 256, 32, 32)

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))   # (B, 128, 64, 64)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 64, 128, 128)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, 32, 256, 256)

        # Output: residual correction added to ThermalGen prediction
        residual = self.out_conv(d1)  # (B, 1, 256, 256)
        output = pred + residual      # Residual learning
        output = torch.clamp(output, 0, 1)

        return output


# ============================================================
# PHASE 2: Train refinement network
# ============================================================
def phase2_train_refinement():
    print("=" * 60)
    print("  PHASE 2: Training refinement network")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check Phase 1 outputs exist
    train_pred_dir = BASELINE_PRED_DIR / "train"
    if not train_pred_dir.exists() or len(os.listdir(train_pred_dir)) == 0:
        print("  ERROR: Run Phase 1 first! No baseline predictions found.")
        sys.exit(1)

    # Load metadata
    print("  Loading metadata...")
    split_data, weather_data = load_metadata()

    # Create dataset
    print("  Creating dataset...")
    dataset = RefinementDataset(
        split_data, weather_data,
        TRAIN_RGB_DIR, TRAIN_THERMAL_DIR,
        train_pred_dir, is_test=False
    )
    print(f"  Dataset size: {len(dataset)} samples")

    # Split into train/val (90/10)
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Train: {n_train} | Val: {n_val}")

    # Model, optimizer, loss
    model = RefinementUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Combined loss: L1 + SSIM-like
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Refinement model: {param_count / 1e6:.2f}M parameters")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float('inf')
    epochs = 50

    print(f"\n  Training for {epochs} epochs...")
    print(f"  {'Epoch':>5s} {'Train Loss':>12s} {'Val Loss':>12s} {'Val PSNR':>10s} {'LR':>10s}")
    print(f"  {'─' * 55}")

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss_sum = 0
        for batch in train_loader:
            rgb, pred, weather, alpha, gt, _ = batch
            rgb, pred, weather, alpha, gt = (
                rgb.to(device), pred.to(device), weather.to(device),
                alpha.to(device), gt.to(device)
            )

            output = model(rgb, pred, weather, alpha)
            loss = l1_loss(output, gt) + 0.1 * mse_loss(output, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        train_loss = train_loss_sum / len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss_sum = 0
        val_psnr_sum = 0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                rgb, pred, weather, alpha, gt, _ = batch
                rgb, pred, weather, alpha, gt = (
                    rgb.to(device), pred.to(device), weather.to(device),
                    alpha.to(device), gt.to(device)
                )

                output = model(rgb, pred, weather, alpha)
                loss = l1_loss(output, gt) + 0.1 * mse_loss(output, gt)
                val_loss_sum += loss.item()

                # PSNR
                mse_val = F.mse_loss(output, gt, reduction='none').mean(dim=[1, 2, 3])
                psnr = 10 * torch.log10(1.0 / (mse_val + 1e-8))
                val_psnr_sum += psnr.sum().item()
                val_count += len(psnr)

        val_loss = val_loss_sum / len(val_loader)
        val_psnr = val_psnr_sum / val_count
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_refinement.pth")
            marker = " ★"
        else:
            marker = ""

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:5d} {train_loss:12.6f} {val_loss:12.6f} {val_psnr:10.2f} {lr:10.6f}{marker}")

    print(f"\n  Training complete! Best val loss: {best_val_loss:.6f}")
    print(f"  Best model saved to: {CHECKPOINT_DIR / 'best_refinement.pth'}")


# ============================================================
# PHASE 3: Generate test predictions + evaluate
# ============================================================
def phase3_generate_and_evaluate():
    print("=" * 60)
    print("  PHASE 3: Generating refined predictions + evaluation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    ckpt_path = CHECKPOINT_DIR / "best_refinement.pth"
    if not ckpt_path.exists():
        print("  ERROR: Run Phase 2 first! No refinement model found.")
        sys.exit(1)

    model = RefinementUNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print("  Loaded refinement model")

    split_data, weather_data = load_metadata()

    # ── Evaluate on training data (to see improvement) ──
    print("\n  Evaluating on training data (baseline vs refined)...")
    import lpips as lpips_lib
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

    lpips_fn = lpips_lib.LPIPS(net='alex').to(device).eval()
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    train_dataset = RefinementDataset(
        split_data, weather_data,
        TRAIN_RGB_DIR, TRAIN_THERMAL_DIR,
        BASELINE_PRED_DIR / "train", is_test=False
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=2)

    # Track baseline and refined metrics
    base_psnr, base_ssim, base_lpips = [], [], []
    ref_psnr, ref_ssim, ref_lpips = [], [], []

    with torch.no_grad():
        for batch in train_loader:
            rgb, pred, weather, alpha, gt, _ = batch
            rgb, pred, weather, alpha, gt = (
                rgb.to(device), pred.to(device), weather.to(device),
                alpha.to(device), gt.to(device)
            )

            refined = model(rgb, pred, weather, alpha)

            for i in range(len(gt)):
                p = pred[i:i+1]
                r = refined[i:i+1]
                g = gt[i:i+1]

                # Baseline metrics
                base_psnr.append(psnr_fn(p, g).item())
                base_ssim.append(ssim_fn(p, g).item())
                p3 = p.repeat(1, 3, 1, 1) * 2 - 1
                g3 = g.repeat(1, 3, 1, 1) * 2 - 1
                base_lpips.append(lpips_fn(p3, g3).item())

                # Refined metrics
                psnr_fn.reset()
                ssim_fn.reset()
                ref_psnr.append(psnr_fn(r, g).item())
                ref_ssim.append(ssim_fn(r, g).item())
                r3 = r.repeat(1, 3, 1, 1) * 2 - 1
                ref_lpips.append(lpips_fn(r3, g3).item())

                psnr_fn.reset()
                ssim_fn.reset()

    print(f"\n  {'─' * 55}")
    print(f"  {'Metric':<12s} {'Baseline':>12s} {'Refined':>12s} {'Change':>12s}")
    print(f"  {'─' * 55}")
    bp, rp = np.mean(base_psnr), np.mean(ref_psnr)
    bs, rs = np.mean(base_ssim), np.mean(ref_ssim)
    bl, rl = np.mean(base_lpips), np.mean(ref_lpips)
    print(f"  {'PSNR ↑':<12s} {bp:12.3f} {rp:12.3f} {rp-bp:+12.3f}")
    print(f"  {'SSIM ↑':<12s} {bs:12.4f} {rs:12.4f} {rs-bs:+12.4f}")
    print(f"  {'LPIPS ↓':<12s} {bl:12.4f} {rl:12.4f} {rl-bl:+12.4f}")
    print(f"  {'─' * 55}")

    # ── Generate test predictions ──
    print(f"\n  Generating refined test predictions...")
    test_pred_dir = BASELINE_PRED_DIR / "test"
    if not test_pred_dir.exists():
        print("  ERROR: Run Phase 1 first! No test baseline predictions found.")
        return

    os.makedirs(TEST_PRED_DIR, exist_ok=True)

    test_dataset = RefinementDataset(
        split_data, weather_data,
        TEST_RGB_DIR, None,
        test_pred_dir, is_test=True
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    count = 0
    with torch.no_grad():
        for batch in test_loader:
            rgb, pred, weather, alpha, fnames = batch
            rgb, pred, weather, alpha = (
                rgb.to(device), pred.to(device), weather.to(device), alpha.to(device)
            )

            refined = model(rgb, pred, weather, alpha)

            for i, fname in enumerate(fnames):
                img = to_pil_image(refined[i].cpu())
                img.save(TEST_PRED_DIR / fname)
                count += 1

    print(f"  Done! {count} test predictions saved to {TEST_PRED_DIR}")
    print(f"\n  ★ Submit the images in {TEST_PRED_DIR} for evaluation!")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3],
                        help="1=baseline, 2=train refinement, 3=predict+evaluate")
    args = parser.parse_args()

    if args.phase == 1:
        phase1_generate_baseline()
    elif args.phase == 2:
        phase2_train_refinement()
    elif args.phase == 3:
        phase3_generate_and_evaluate()
