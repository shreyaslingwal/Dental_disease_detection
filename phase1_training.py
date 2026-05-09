

import warnings
warnings.filterwarnings("ignore")
# 1. Import Libraries
import json
import os
import copy
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import average_precision_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 2. CONFIG Dictionary
# --------------------------------------------------
CONFIG = {
    "seed": 42,
    "img_size": 320,          
    "batch_size": 16,         
    "num_workers": 4,
    "num_epochs": 30,
    "learning_rate": 1e-5,
    "backbone_lr": 5e-6,
    "weight_decay": 1e-4,
    "early_stopping_patience": 7,
    "num_classes": 4,         
    "model_name": "swin_small_patch4_window7_224", 
    "dropout_rate": 0.2,      
    "focal_gamma": 1.0,        
    "label_smoothing": 0.05,
    # Per-class decision thresholds [Caries, Deep Caries, Impacted, Periapical]
    "class_thresholds": [0.6, 0.5, 0.5, 0.5],
 
    "bbox_pad_ratio": 0.30,
    # Minimum crop size in pixels (skip tiny/corrupted bboxes)
    "min_crop_size": 20,
    # Dataset path on Kaggle
    "base_path": "/kaggle/input/datasets/truthisneverlinear/dentex-challenge-2023/training_data/training_data/quadrant-enumeration-disease",
    "output_path": "/kaggle/working/",
    # Phase 0 pre-trained backbone ( if available)
    "pretrained_backbone_paths": [
        "/kaggle/working/swin_small_pretrained_backbone.pth",
        "/kaggle/input/dental-pretrained/swin_small_pretrained_backbone.pth",
    ],
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CONFIG["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------
# 3. Load and Parse DENTEX Annotations (Per-Tooth Crops)
# --------------------------------------------------

json_file = os.path.join(CONFIG["base_path"], "train_quadrant_enumeration_disease.json")

with open(json_file, "r") as f:
    data = json.load(f)

# Build image lookups
id_to_img_info = {img["id"]: img for img in data["images"]}
id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

# Disease categories from category_id_3
disease_categories = {cat["id"]: cat["name"] for cat in data["categories_3"]}
print(f"Disease categories: {disease_categories}")

disease_name_to_idx = {name: idx for idx, name in enumerate(sorted(disease_categories.values()))}
print(f"Disease label mapping: {disease_name_to_idx}")

# Build per-tooth crop samples: (image_path, bbox, disease_idx)

img_dir = os.path.join(CONFIG["base_path"], "xrays")
crop_samples = []  

for ann in data["annotations"]:
    img_id = ann["image_id"]
    filename = id_to_filename.get(img_id)
    if filename is None:
        continue

    img_path = os.path.join(img_dir, filename)
    if not os.path.exists(img_path):
        continue

    # Get bbox [x, y, w, h] in COCO format
    bbox = ann.get("bbox", None)
    if bbox is None:
        continue

    x, y, w, h = bbox
    if w < CONFIG["min_crop_size"] or h < CONFIG["min_crop_size"]:
        continue

    # Disease label (single-label per tooth -> class index)
    disease_id = ann["category_id_3"]
    disease_name = disease_categories[disease_id]
    label_idx = disease_name_to_idx[disease_name]

    crop_samples.append({
        "img_path": img_path,
        "bbox": [x, y, w, h],
        "label": label_idx,  # Single integer class index
        "disease_name": disease_name,
        "img_id": img_id,
    })

# --------------------------------------------------
# 3b. Disease-Only Tooth Crops
# --------------------------------------------------
# DENTEX category_id_3 contains the 4 pathology classes only.
from tqdm import tqdm

print("Training on DENTEX disease-only tooth crops (4 classes).")

# Count per-disease distribution
disease_dist = Counter(s["disease_name"] for s in crop_samples)
print(f"\nTotal DISEASE tooth crops: {len(crop_samples)}")
print(f"From {len(set(s['img_id'] for s in crop_samples))} unique images")
print(f"Per-disease distribution:")
for name in sorted(disease_dist.keys()):
    print(f"  {name}: {disease_dist[name]}")

# --------------------------------------------------
# 4. Compute Class Weights for CrossEntropyLoss
# --------------------------------------------------
all_labels = np.array([s["label"] for s in crop_samples])
N = len(crop_samples)
class_counts = np.bincount(all_labels, minlength=CONFIG["num_classes"])

# Manual class weights for DENTEX 4 classes:
# [Caries, Deep Caries, Impacted, Periapical Lesion]
CONFIG["class_weights"] = [0.7, 2.5, 2.5, 5.0]

class_names = sorted(disease_name_to_idx.keys())
print(f"\nClass distribution:")
for i, name in enumerate(class_names):
    print(f"  {name}: {class_counts[i]} samples")
print(f"Manual class_weights: {CONFIG['class_weights']}")

# --------------------------------------------------
# 5. EDA - Visualize Disease Distribution
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.bar(class_names, class_counts, color="skyblue", edgecolor="black")
plt.title("Distribution of Tooth-Level Disease Labels (Crops)")
plt.ylabel("Number of Tooth Crops")
plt.xlabel("Disease")
for i, v in enumerate(class_counts):
    plt.text(i, v + 1, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_path"], "disease_distribution.png"), dpi=150)
plt.show()

# --------------------------------------------------
# 6. Train-Validation Split (Image-Level to Prevent Leakage)
# --------------------------------------------------
# Split by IMAGE, not by crop. All crops from one image go to same split.
# This prevents data leakage where the model sees different teeth from the
# same X-ray in both train and val.
unique_img_ids = list(set(s["img_id"] for s in crop_samples))
train_img_ids, val_img_ids = train_test_split(
    unique_img_ids,
    test_size=0.2,
    random_state=CONFIG["seed"],
)
train_img_ids_set = set(train_img_ids)
val_img_ids_set = set(val_img_ids)

train_crops = [s for s in crop_samples if s["img_id"] in train_img_ids_set]
val_crops = [s for s in crop_samples if s["img_id"] in val_img_ids_set]

print(f"\nImage-level split:")
print(f"  Train: {len(train_img_ids)} images -> {len(train_crops)} crops")
print(f"  Val:   {len(val_img_ids)} images -> {len(val_crops)} crops")

# Save val split manifest for Phase 2 (grouped by image for two-stage pipeline)
# Phase 2 needs: for each image, all tooth bboxes + their disease labels
val_manifest_by_image = {}
for s in val_crops:
    img_p = s["img_path"]
    if img_p not in val_manifest_by_image:
        val_manifest_by_image[img_p] = {
            "img_path": img_p,
            "img_id": s["img_id"],
            "teeth": [],
        }
    val_manifest_by_image[img_p]["teeth"].append({
        "bbox": s["bbox"],
        "labels": s["label"],
        "disease_name": s["disease_name"],
    })

val_manifest = list(val_manifest_by_image.values())
val_manifest_path = os.path.join(CONFIG["output_path"], "val_split_manifest.json")
with open(val_manifest_path, "w") as f:
    json.dump(val_manifest, f, indent=2)
print(f"Val split manifest saved to: {val_manifest_path}")
print(f"  {len(val_manifest)} images, {len(val_crops)} total tooth crops")

# --------------------------------------------------
# 7. Albumentations Augmentations (for crops)
# --------------------------------------------------
# Different augmentation style from primary Swin-T (ensemble diversity)
# Focus: robustness-building transforms (dropout, blur) instead of pixel-level (CLAHE, noise)
train_transform = A.Compose([
    A.Resize(CONFIG["img_size"], CONFIG["img_size"]),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(CONFIG["img_size"], CONFIG["img_size"]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# --------------------------------------------------
# 8a. Pre-load ALL Images into RAM (runs once)
# --------------------------------------------------
from tqdm import tqdm

unique_img_paths = sorted(set(s["img_path"] for s in crop_samples))
image_cache = {}  # Shared between train and val datasets
print(f"Pre-loading {len(unique_img_paths)} unique images into RAM...")
for img_path in tqdm(unique_img_paths, desc="Loading images"):
    image = cv2.imread(img_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_cache[img_path] = image
print(f"Cached {len(image_cache)} images")

# --------------------------------------------------
# 8b. Tooth Crop Dataset
# --------------------------------------------------
class ToothCropDataset(Dataset):
    """Dataset that extracts tooth crops from full X-rays using bboxes.

    Each sample is a single tooth crop with its disease label.
    Crops are extracted with padding for context around the tooth.
    Images are pre-loaded into a shared cache passed at init.
    """
    def __init__(self, crop_list, image_cache, transform=None, bbox_pad_ratio=0.15):
        self.crops = crop_list
        self.image_cache = image_cache
        self.transform = transform
        self.bbox_pad_ratio = bbox_pad_ratio

    def _extract_crop(self, image, bbox):
        """Extract a padded crop from the image."""
        img_h, img_w = image.shape[:2]
        x, y, w, h = bbox

        # Add padding for context
        pad_w = int(w * self.bbox_pad_ratio)
        pad_h = int(h * self.bbox_pad_ratio)

        x1 = max(0, int(x) - pad_w)
        y1 = max(0, int(y) - pad_h)
        x2 = min(img_w, int(x + w) + pad_w)
        y2 = min(img_h, int(y + h) + pad_h)

        crop = image[y1:y2, x1:x2]

        # Safety: ensure valid crop
        if crop.shape[0] < 5 or crop.shape[1] < 5:
            # Fallback: return center crop of the image
            cx, cy = img_w // 2, img_h // 2
            half = min(img_w, img_h) // 4
            crop = image[cy-half:cy+half, cx-half:cx+half]

        return crop

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        sample = self.crops[idx]
        image = self.image_cache.get(sample["img_path"])

        if image is None:
            # Return a black image as fallback
            crop = np.zeros((CONFIG["img_size"], CONFIG["img_size"], 3), dtype=np.uint8)
        else:
            crop = self._extract_crop(image, sample["bbox"])

        if self.transform:
            augmented = self.transform(image=crop)
            crop = augmented["image"]

        label = torch.tensor(sample["label"], dtype=torch.long)
        return crop, label


train_dataset = ToothCropDataset(train_crops, image_cache, transform=train_transform, bbox_pad_ratio=CONFIG["bbox_pad_ratio"])
val_dataset = ToothCropDataset(val_crops, image_cache, transform=val_transform, bbox_pad_ratio=CONFIG["bbox_pad_ratio"])


print(f"\nUsing natural distribution (no sampler, shuffle=True):")
train_labels = np.array([s["label"] for s in train_crops])
train_class_counts = np.bincount(train_labels, minlength=CONFIG["num_classes"])
for i, name in enumerate(class_names):
    print(f"  {name}: count={train_class_counts[i]}")

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,  # Natural distribution instead of oversampling
    num_workers=CONFIG["num_workers"],
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["num_workers"],
    pin_memory=True,
)

# Sanity check
batch_imgs, batch_labels = next(iter(train_loader))
print(f"\nBatch shape: {batch_imgs.shape} | Labels shape: {batch_labels.shape}")
print(f"Labels in batch: {torch.bincount(batch_labels, minlength=CONFIG['num_classes']).tolist()} (per-class counts)")

# --------------------------------------------------
# 9. Model Definition - Swin-Tiny with CSRA + Independent Binary Heads
# --------------------------------------------------
class CSRA(nn.Module):
    """Class-Specific Residual Attention (inspired by DENTEX 4th place).
    Each attention head focuses on different spatial regions per class.
    Uses multi-temperature spatial softmax for diverse attention patterns."""
    def __init__(self, in_dim, num_classes, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_classes = num_classes
        # Each head: 1x1 conv producing class-specific attention scores
        self.attn_convs = nn.ModuleList([
            nn.Conv2d(in_dim, num_classes, kernel_size=1)
            for _ in range(num_heads)
        ])
        # Temperatures: sharper to broader attention across heads
        temps = [1.0, 2.0, 4.0, 8.0][:num_heads]
        self.register_buffer("temps", torch.tensor(temps).float())
        # Final projection: class-specific pooled features -> 1 logit per class
        self.projections = nn.ModuleList([
            nn.Linear(in_dim, 1) for _ in range(num_classes)
        ])

    def forward(self, feature_map):
        """Args: feature_map (B, C, H, W) -> returns logits (B, num_classes)"""
        B, C, H, W = feature_map.shape
        # Accumulate class-specific pooled features across heads
        class_features = torch.zeros(B, self.num_classes, C,
                                     device=feature_map.device)
        for head_conv, temp in zip(self.attn_convs, self.temps):
            # Class attention scores: (B, K, H, W)
            score = head_conv(feature_map)
            # Spatial softmax with temperature: (B, K, H*W)
            attn = F.softmax(score.view(B, self.num_classes, -1) / temp, dim=-1)
            # Weighted spatial pooling per class:
            # feature_map: (B, C, H*W), attn: (B, K, H*W)
            feat_flat = feature_map.view(B, C, -1)  # (B, C, H*W)
            # For each class k, pool = sum(attn[k] * features) over spatial
            # (B, C, H*W) x (B, K, H*W)^T -> (B, C, K) via einsum
            pooled = torch.einsum('bcn,bkn->bkc', feat_flat, attn)  # (B, K, C)
            class_features += pooled

        class_features = class_features / self.num_heads  # Average across heads

        # Project each class's pooled features to 1 logit
        logits = []
        for k in range(self.num_classes):
            logits.append(self.projections[k](class_features[:, k, :]))  # (B, 1)
        return torch.cat(logits, dim=1)  # (B, num_classes)


class DentalCNN(nn.Module):
    """Swin-Tiny with CSRA attention + 4 independent binary disease heads.

    Architecture (DENTEX-inspired):
      1. Swin-T backbone (global self-attention, all top-3 DENTEX teams used transformers)
      2. CSRA module (class-specific spatial attention on feature maps)
      3. 4 independent binary heads (each head learns disease-specific features)
      4. Residual connection: GAP features + CSRA features combined

    Input: cropped tooth image (224x224)
    Output: 4 independent binary logits (one per disease)
    """
    def __init__(self, model_name, num_classes, pretrained=True, dropout_rate=0.3, img_size=224):
        super().__init__()
        # Swin backbone WITHOUT global pooling (keep feature maps)
        # img_size tells timm to adapt window/position embeddings for non-native resolution
        self.backbone = timm.create_model(model_name, pretrained=pretrained,
                                          num_classes=0, global_pool="",
                                          img_size=img_size)
        self.backbone.patch_embed.strict_img_size = False  # Allow flexible input sizes
        backbone_dim = self.backbone.num_features
        self.num_classes = num_classes
        self.dropout = nn.Dropout(p=dropout_rate)

        # Global average pooling (fallback path)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # CSRA attention module (class-specific spatial pooling)
        self.csra = CSRA(backbone_dim, num_classes, num_heads=4)

        # 4 independent binary classification heads
        self.heads = nn.ModuleList([
            nn.Linear(backbone_dim, 1) for _ in range(num_classes)
        ])

    def forward(self, x):
        feature_map = self.backbone(x)  # Swin-T outputs (B, H, W, C) channels-last
        feature_map = feature_map.permute(0, 3, 1, 2)  # -> (B, C, H, W) channels-first

        # Path 1: GAP features -> independent heads
        gap_features = self.gap(feature_map).flatten(1)  # (B, C)
        gap_features = self.dropout(gap_features)
        head_logits = torch.cat([h(gap_features) for h in self.heads], dim=1)  # (B, K)

        # Path 2: CSRA attention logits
        csra_logits = self.csra(feature_map)  # (B, K)

        # Residual combination: head output + CSRA attention
        return head_logits + csra_logits  # (B, num_classes)

# Initialize model (pretrained=True for ImageNet weights from timm)
model = DentalCNN(
    model_name=CONFIG["model_name"],
    num_classes=CONFIG["num_classes"],
    pretrained=True,
    dropout_rate=CONFIG["dropout_rate"],
    img_size=CONFIG["img_size"],        # 320px (non-native, timm will adapt)
).to(device)

# Try loading Phase 0 pre-trained backbone (if a Swin-Small version exists)
pretrained_path = None
for candidate in CONFIG.get("pretrained_backbone_paths", []):
    if os.path.exists(candidate):
        pretrained_path = candidate
        break

if pretrained_path:
    print(f"Found Phase 0 backbone at: {pretrained_path}")
    pretrained_state = torch.load(pretrained_path, map_location=device, weights_only=True)
    missing, unexpected = model.backbone.load_state_dict(pretrained_state, strict=False)
    print(f"Loaded Phase 0 pre-trained backbone")
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if len(missing) > 10:
        print("  WARNING: Many missing keys - architecture mismatch. Using ImageNet weights instead.")
else:
    print("No Phase 0 Swin-Small backbone found (expected - using ImageNet pretrained weights)")
    print(f"ImageNet-pretrained {CONFIG['model_name']} loaded successfully")

# Minimal freeze: ONLY patch_embed frozen, ALL Swin layers trainable

for name, param in model.backbone.named_parameters():
    if name.startswith("patch_embed"):  # freeze only patch embedding (stem)
        param.requires_grad = False

frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = frozen + trainable
print(f"Minimal freeze (only patch_embed frozen, all Swin layers trainable):")
print(f"  Frozen: {frozen:,} | Trainable: {trainable:,} | Total: {total:,} ({100*trainable/total:.1f}% trainable)")

# Verify forward pass
dummy_input = torch.randn(1, 3, CONFIG["img_size"], CONFIG["img_size"]).to(device)
dummy_output = model(dummy_input)
print(f"Model output shape: {dummy_output.shape}")  # Should be (1, 4)

# --------------------------------------------------
# 10. Focal CrossEntropy Loss + Optimizer + Scheduler + Scaler
# --------------------------------------------------.
class FocalCrossEntropyLoss(nn.Module):
    """Focal loss on top of CrossEntropy for single-label classification.
    Reduces contribution of easy examples, focuses on hard ones.
    gamma=1.0 for smoother focus (ensemble diversity)."""
    def __init__(self, gamma=1.0, class_weights=None, num_classes=4):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        if class_weights is not None:
            self.register_buffer("weight", class_weights)
        else:
            self.weight = None

    def forward(self, logits, targets):
        # Standard CE with class weights
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        # Focal modulation: reduce easy-sample contribution
        p_t = torch.exp(-ce_loss)  # probability of correct class
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()

# Class weights for imbalance handling
class_weights = torch.tensor(CONFIG["class_weights"], dtype=torch.float32).to(device)
print(f"Class weights (CE): {CONFIG['class_weights']}")
print(f"Focal gamma: {CONFIG['focal_gamma']}")
criterion = FocalCrossEntropyLoss(
    gamma=CONFIG["focal_gamma"],
    class_weights=class_weights,
    num_classes=CONFIG["num_classes"],
)

# Differential LR: lower for unfrozen backbone layers, higher for heads
head_params = list(model.heads.parameters()) + list(model.csra.parameters())
head_ids = [id(p) for p in head_params]
backbone_trainable = [p for p in model.backbone.parameters() if p.requires_grad and id(p) not in head_ids]

optimizer = torch.optim.AdamW([
    {"params": backbone_trainable, "lr": CONFIG["backbone_lr"]},
    {"params": head_params, "lr": CONFIG["learning_rate"]},
], weight_decay=CONFIG["weight_decay"])

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=CONFIG["num_epochs"],
    eta_min=1e-6,
)

scaler = GradScaler()

# --------------------------------------------------
# 11. Training and Validation Functions
# --------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()  # Softmax (consistent with CE loss)
            all_preds.append(probs)
            all_targets.append(labels.cpu().numpy())  # Integer class indices

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.vstack(all_preds)
    all_targets = np.concatenate(all_targets)

    # Top-1 accuracy
    pred_classes = all_preds.argmax(axis=1)
    accuracy = accuracy_score(all_targets, pred_classes)

    # mAP (convert integer labels to one-hot for average_precision_score)
    num_classes = all_preds.shape[1]
    targets_onehot = np.eye(num_classes)[all_targets]
    mAP = average_precision_score(targets_onehot, all_preds, average="macro")

    return epoch_loss, accuracy, mAP, all_preds, all_targets

# --------------------------------------------------
# 12. Training Loop with Early Stopping
# --------------------------------------------------
best_mAP = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
patience_counter = 0
train_losses = []
val_losses = []
val_accs = []
val_maps = []

print("\nStarting tooth-level training (single-label CE, two-stage pipeline)... [ignoring loop detection]")
print(f"Train crops: {len(train_crops)} | Val crops: {len(val_crops)}")
print(f"Epochs: {CONFIG['num_epochs']} | Batch: {CONFIG['batch_size']} | Img: {CONFIG['img_size']}px")
print(f"Loss: FocalCrossEntropy (softmax) | gamma={CONFIG['focal_gamma']}")
print(f"Checkpointing on: mAP (best so far)")
print("-" * 60)

for epoch in range(CONFIG["num_epochs"]):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
    val_loss, val_acc, val_mAP, _, _ = validate(model, val_loader, criterion)
    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_maps.append(val_mAP)

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] "
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | Val mAP: {val_mAP:.4f} | LR: {current_lr:.6f}"
    )

    # Checkpoint on best mAP
    if val_mAP > best_mAP:
        best_mAP = val_mAP
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0
        torch.save(
            model.state_dict(),
            os.path.join(CONFIG["output_path"], "swin_s_dental_best.pth"),
        )
        print(f"  -> New best model saved (mAP: {best_mAP:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= CONFIG["early_stopping_patience"]:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

print("-" * 60)
print(f"Training complete. Best Val mAP: {best_mAP:.4f}")

# --------------------------------------------------
# 13. Training Curves
# --------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(train_losses, label="Train Loss")
axes[0].plot(val_losses, label="Val Loss")
axes[0].set_title("Loss Curves (Tooth-Level)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(val_accs, label="Val Accuracy", color="green")
axes[1].set_title("Validation Accuracy (Tooth-Level)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(val_maps, label="Val mAP", color="orange")
axes[2].set_title("Validation mAP (Tooth-Level)")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("mAP")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_path"], "training_curves.png"), dpi=150)
plt.show()

# --------------------------------------------------
# 14. Final Evaluation on Best Model
# --------------------------------------------------
model.load_state_dict(best_model_wts)
val_loss, val_acc, val_mAP, all_preds, all_targets = validate(model, val_loader, criterion)

# Top-1 accuracy
pred_classes = all_preds.argmax(axis=1)
print(f"\nFinal Val Accuracy: {val_acc:.4f} | Final Val mAP: {val_mAP:.4f}")

# Full classification report
print(f"\nClassification Report (single-label, argmax):")
print(classification_report(all_targets, pred_classes, target_names=class_names, zero_division=0))

# Per-class sigmoid confidence stats
print("Per-class mean confidence (sigmoid):")
for i, name in enumerate(class_names):
    mask = all_targets == i
    if mask.sum() > 0:
        mean_conf = all_preds[mask, i].mean()
        print(f"  {name}: {mean_conf:.4f} (n={mask.sum()})")
    else:
        print(f"  {name}: no samples")

# --------------------------------------------------
# 15. Save Final Checkpoint
# --------------------------------------------------
torch.save(
    {
        "model_state_dict": best_model_wts,
        "config": CONFIG,
        "class_names": class_names,
        "disease_name_to_idx": disease_name_to_idx,
        "best_mAP": best_mAP,
        "loss_type": "FocalCrossEntropy",
        "architecture": "swin_s_csra_softmax_heads",
        "pipeline": "two_stage_tooth_crop",
        "bbox_pad_ratio": CONFIG["bbox_pad_ratio"],
        "img_size": CONFIG["img_size"],
    },
    os.path.join(CONFIG["output_path"], "swin_s_dental_full_checkpoint.pth"),
)
print(f"\nFull checkpoint saved to {CONFIG['output_path']}swin_s_dental_full_checkpoint.pth")

