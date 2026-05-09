
# --------------------------------------------------
# 1. Import Libraries
# --------------------------------------------------
import os
import json
import glob
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------
# 2. CONFIG
# --------------------------------------------------
CONFIG = {
    "seed": 99,
    "img_size": 336,          
    "num_classes": 4,
    "model_name": "swin_small_patch4_window7_224",  
    "dropout_rate": 0.4,      
    # Disease class mapping (sorted alphabetically, same as Phase 1)
    "disease_names": {
        0: "Caries",
        1: "Deep Caries",
        2: "Impacted",
        3: "Periapical Lesion",
    },
    # GradCAM settings 
    "cam_target_layer": "layers.3.blocks.1.norm2",
    "cam_colormap": cv2.COLORMAP_JET,
    "generate_gradcam_overlays": True,  
    # Legacy crop-classifier thresholds 
    "model_conf_threshold": 0.60,
    "uncertain_threshold": 0.4,
    "margin_threshold": 0.15,
    "temperature": 1.5,
    "top_k": 2,
   
    "class_thresholds": [0.60, 0.55, 0.50, 0.50],

    "class_threshold_adjustments": {
        "Caries": 0.0,          
        "Deep Caries": 0.0,       
        "Impacted": 0.0,         
        "Periapical Lesion": -0.05,  
    },
   
    "tooth_crop_min_conf_threshold": 0.65,

    "wbf_final_min_conf_threshold": 0.60,

    "uncertain_margin": 0.1,
    "caries_suppression_threshold": 0.60,
   
    "max_findings_per_tooth": 2,
    "cam_intensity_threshold": 0.05,
    "max_findings": 12,
    "tta_enabled": True,
    # -----------------------------------------------
    # WBF Ensemble 
    # -----------------------------------------------
    "use_wbf_ensemble": True,         
    "wbf_iou_threshold": 0.55,        
    "wbf_skip_box_thr": 0.3,          
    "wbf_weights": [1.0, 1.0],         
    # avg = requires detector agreement 
    "wbf_conf_type": "avg",
    "tooth_match_iou_threshold": 0.40, 
    # YOLO disease detector 
    "yolo_disease_model_paths": [
        "/kaggle/input/datasets/shreyas2123/latest-best/yolo_disease_best.pt",
        "/kaggle/working/yolo_disease_best.pt",
        "/kaggle/input/dental-yolo/yolo_disease_best.pt",
    ],
    "yolo_disease_conf_threshold": 0.5, 
    # Disease class names for YOLO disease detector 
    "disease_class_names_yolo": ["Caries", "Deep Caries", "Impacted", "Periapical Lesion"],
    # Phase 1 crop classifier checkpoint 
    "checkpoint_paths": [
        "/kaggle/input/datasets/shreyas2123/latest-best/swin_s_dental_best_82.pth",
        "/kaggle/input/dental-phase1/swin_s_dental_full_checkpoint.pth",
    ],
    # YOLO tooth detection model 
    "use_yolo_for_eval": True,
    "yolo_conf_threshold": 0.45,
    "yolo_iou_threshold": 0.6,
    "yolo_model_paths": [
        "/kaggle/input/datasets/shreyas2123/latest-best/yolo_tooth_best.pt",
        "/kaggle/working/yolo_tooth_best.pt",
        "/kaggle/working/yolo_tooth_detection/train/weights/best.pt",
        "/kaggle/input/dental-yolo/yolo_tooth_best.pt",
    ],
    # Data paths
    "base_path": "/kaggle/input/datasets/truthisneverlinear/dentex-challenge-2023/training_data/training_data/quadrant-enumeration-disease",
    "output_path": "/kaggle/working/",
    "val_manifest_paths": [
        "/kaggle/working/val_split_manifest.json",
        "/kaggle/input/dental-phase1/val_split_manifest.json",
    ],
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------
# 3. Model Definition 
# --------------------------------------------------
class CSRA(nn.Module):
    """Class-Specific Residual Attention (inspired by DENTEX 4th place).
    Each attention head focuses on different spatial regions per class."""
    def __init__(self, in_dim, num_classes, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.attn_convs = nn.ModuleList([
            nn.Conv2d(in_dim, num_classes, kernel_size=1)
            for _ in range(num_heads)
        ])
        temps = [1.0, 2.0, 4.0, 8.0][:num_heads]
        self.register_buffer("temps", torch.tensor(temps).float())
        self.projections = nn.ModuleList([
            nn.Linear(in_dim, 1) for _ in range(num_classes)
        ])

    def forward(self, feature_map):
        B, C, H, W = feature_map.shape
        class_features = torch.zeros(B, self.num_classes, C,
                                     device=feature_map.device)
        for head_conv, temp in zip(self.attn_convs, self.temps):
            score = head_conv(feature_map)
            attn = F.softmax(score.view(B, self.num_classes, -1) / temp, dim=-1)
            feat_flat = feature_map.view(B, C, -1)
            pooled = torch.einsum('bcn,bkn->bkc', feat_flat, attn)
            class_features += pooled
        class_features = class_features / self.num_heads
        logits = []
        for k in range(self.num_classes):
            logits.append(self.projections[k](class_features[:, k, :]))
        return torch.cat(logits, dim=1)


class DentalCNN(nn.Module):
    """Swin-Small with independent binary heads + CSRA attention.
    Must match the exact architecture used during Phase 1 training.
    NOTE: Update this if loading a checkpoint from a different Phase 1 variant."""
    def __init__(self, model_name, num_classes, pretrained=False, dropout_rate=0.3, img_size=336):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained,
                                          num_classes=0, global_pool="",
                                          img_size=img_size)
        self.backbone.patch_embed.strict_img_size = False
        backbone_dim = self.backbone.num_features
        self.num_classes = num_classes
        self.dropout = nn.Dropout(p=dropout_rate)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.csra = CSRA(backbone_dim, num_classes, num_heads=4)
        # Independent binary heads (matches old Phase 1 checkpoint)
        self.heads = nn.ModuleList([
            nn.Linear(backbone_dim, 1) for _ in range(num_classes)
        ])

    def forward(self, x):
        feature_map = self.backbone(x)  # Swin-S outputs (B, H, W, C) channels-last
        feature_map = feature_map.permute(0, 3, 1, 2)  # -> (B, C, H, W) channels-first
        gap_features = self.gap(feature_map).flatten(1)
        gap_features = self.dropout(gap_features)
        head_logits = torch.cat([h(gap_features) for h in self.heads], dim=1)
        csra_logits = self.csra(feature_map)
        return head_logits + csra_logits

# --------------------------------------------------
# 4. Load Trained Phase 1 Checkpoint
# --------------------------------------------------
def load_phase1_model(config):
    """Load the best Phase 1 trained model from available checkpoint locations."""
    model = DentalCNN(
        model_name=config["model_name"],
        num_classes=config["num_classes"],
        pretrained=False,
        dropout_rate=config["dropout_rate"],
        img_size=config["img_size"],
    ).to(device)

    checkpoint_path = None
    for candidate in config["checkpoint_paths"]:
        if os.path.exists(candidate):
            checkpoint_path = candidate
            break

    if checkpoint_path is None:
        raise FileNotFoundError(
            "Phase 1 checkpoint not found. Searched:\n"
            + "\n".join(f"  - {p}" for p in config["checkpoint_paths"])
        )

    print(f"Loading Phase 1 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both full checkpoint dict and raw state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Best mAP from training: {checkpoint.get('best_mAP', 'N/A')}")
        print(f"  Class names: {checkpoint.get('class_names', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
        print("  Loaded raw state_dict (no metadata)")

    model.eval()
    print("Phase 1 model loaded and set to eval mode.")
    return model

model = load_phase1_model(CONFIG)

# --------------------------------------------------
# 5. Preprocessing (matches Phase 1 val_transform)
# --------------------------------------------------
val_transform = A.Compose([
    A.Resize(CONFIG["img_size"], CONFIG["img_size"]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def preprocess_image(img_path):
    """Load an image and return (tensor, original_rgb_image)."""
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize original for overlay visualization (before normalization)
    display_img = cv2.resize(image_rgb, (CONFIG["img_size"], CONFIG["img_size"]))

    # Apply val transform for model input
    augmented = val_transform(image=image_rgb)
    tensor = augmented["image"].unsqueeze(0).to(device)  # (1, 3, H, W)

    return tensor, display_img


# ==================================================
# 6. GradCAM Implementation
# ==================================================
class GradCAM:
    """Gradient-weighted Class Activation Mapping for Swin-T backbone.

    Uses register hooks on the target layer to capture:
      - Forward pass: feature map activations (spatial)
      - Backward pass: gradients flowing through the layer

    Then computes per-class heatmaps by weighting activations with
    global-average-pooled gradients.
    """
    def __init__(self, model, target_layer_name="conv_head"):
        self.model = model
        self.activations = None
        self.gradients = None
        self._hooks = []

        # Resolve the target layer inside the backbone
        target_layer = dict(model.backbone.named_modules()).get(target_layer_name)
        if target_layer is None:
            raise ValueError(
                f"Layer '{target_layer_name}' not found in model.backbone. "
                f"Available: {[n for n, _ in model.backbone.named_modules() if n]}"
            )

        # Register forward hook to capture activations
        self._hooks.append(
            target_layer.register_forward_hook(self._forward_hook)
        )
        # Register backward hook to capture gradients
        self._hooks.append(
            target_layer.register_full_backward_hook(self._backward_hook)
        )

        print(f"GradCAM initialized on layer: backbone.{target_layer_name}")

    def _forward_hook(self, module, input, output):
        """Capture feature map activations during forward pass."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Capture gradients during backward pass."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        """Generate GradCAM heatmap for a specific class.

        Args:
            input_tensor: (1, 3, H, W) preprocessed image tensor
            class_idx: integer class index to explain

        Returns:
            heatmap: (H, W) numpy array normalized to [0, 1]
        """
        self.model.eval()

        # Forward pass - hooks capture activations
        # Need gradients enabled for backward pass
        input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)

        # Zero all existing gradients
        self.model.zero_grad()

        # Backward pass for target class - hooks capture gradients
        target_score = logits[0, class_idx]
        target_score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients. Check target layer.")

        activations = self.activations
        gradients = self.gradients

        # Swin-T norm layers output (B, N, C) tokens instead of (B, C, H, W)
        # Reshape to spatial format for GradCAM computation
        if activations.dim() == 3:
            B, N, C = activations.shape
            h = w = int(N ** 0.5)  # 49 tokens -> 7x7 spatial
            activations = activations.permute(0, 2, 1).reshape(B, C, h, w)
            gradients = gradients.permute(0, 2, 1).reshape(B, C, h, w)

        # Global Average Pool the gradients -> channel weights (1, C, 1, 1)
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)

        # ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Squeeze to (h, w) and move to CPU
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input image size
        cam_resized = cv2.resize(cam, (CONFIG["img_size"], CONFIG["img_size"]))

        return cam_resized

    def compute_cam_intensity(self, cam):
        """Compute meaningful GradCAM intensity using top-10% mean.

        Using max() always gives ~0.99 because the heatmap is normalized.
        Instead, compute mean of the top 10% of pixel values as a proxy
        for how spatially focused the activation is.

        Args:
            cam: (H, W) numpy heatmap normalized to [0, 1]

        Returns:
            float: intensity score in [0, 1]
        """
        flat = cam.flatten()
        if flat.max() == 0:
            return 0.0
        # Top 10% threshold
        top_k = max(1, int(len(flat) * 0.10))
        top_vals = np.sort(flat)[-top_k:]
        return float(np.mean(top_vals))

    def generate_all_classes(self, input_tensor, probabilities):
        """Generate heatmaps for all classes that exceed the confidence threshold.

        Args:
            input_tensor: (1, 3, H, W) preprocessed image tensor
            probabilities: (num_classes,) numpy array of softmax probabilities

        Returns:
            dict mapping class_idx -> heatmap (H, W) numpy array
        """
        heatmaps = {}
        for cls_idx in range(len(probabilities)):
            if probabilities[cls_idx] > CONFIG["model_conf_threshold"]:
                heatmap = self.generate(input_tensor, cls_idx)
                heatmaps[cls_idx] = heatmap
        return heatmaps

    def cleanup(self):
        """Remove hooks to free memory."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


gradcam = GradCAM(model, target_layer_name=CONFIG["cam_target_layer"])

# --------------------------------------------------
# 7. Heatmap Thresholding + Contour Detection
# --------------------------------------------------
def threshold_heatmap(heatmap):
    """Binarize GradCAM heatmap using Otsu thresholding and find disease regions.

    Args:
        heatmap: (H, W) float array in [0, 1]

    Returns:
        contours: list of contour arrays
        binary_map: (H, W) binary uint8 image
        centers: list of (cx, cy) centroid tuples
        bboxes: list of (x, y, w, h) bounding box tuples
    """
    # Convert to uint8 for OpenCV
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Otsu thresholding to auto-find the optimal binarization cutoff
    _, binary_map = cv2.threshold(
        heatmap_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Find contours of disease islands
    contours, _ = cv2.findContours(
        binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    centers = []
    bboxes = []
    min_area = (CONFIG["img_size"] * CONFIG["img_size"]) * 0.005  # Ignore tiny noise (<0.5% of image)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # Bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, w, h))

        # Centroid via moments
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
        centers.append((cx, cy))

    return contours, binary_map, centers, bboxes


# ==================================================
# 8. FDI Quadrant Anatomical Mapping
# ==================================================
def map_to_quadrant(x, y, w=512, h=512):
    """Map pixel coordinates to FDI dental quadrant (1-4).

    Panoramic X-ray orientation (standard):
      - Patient's RIGHT side appears on the LEFT of the image
      - Upper jaw (maxillary) is the top half
      - Lower jaw (mandibular) is the bottom half

    FDI Quadrant Layout (from patient's perspective):
      Q1 (upper right) | Q2 (upper left)
      -----------------+-----------------
      Q4 (lower right) | Q3 (lower left)

    On the radiograph (mirrored):
      Image LEFT  = Patient RIGHT (Q1 upper, Q4 lower)
      Image RIGHT = Patient LEFT  (Q2 upper, Q3 lower)

    Args:
        x, y: pixel coordinates on the (w, h) image
        w, h: image dimensions (default 512x512)

    Returns:
        quadrant: integer 1, 2, 3, or 4
    """
    is_patient_right = x < w / 2   # Left side of image = patient's right
    is_upper = y < h / 2           # Top half = maxillary (upper jaw)

    if is_patient_right and is_upper:
        return 1   # Upper right (maxillary right)
    elif not is_patient_right and is_upper:
        return 2   # Upper left (maxillary left)
    elif not is_patient_right and not is_upper:
        return 3   # Lower left (mandibular left)
    else:
        return 4   # Lower right (mandibular right)


# ==================================================
# 9. FDI Tooth Enumeration
# ==================================================
def extract_fdi_tooth(x, quadrant, w=512):
    """Map x-coordinate + quadrant to a specific FDI tooth number.

    Each quadrant has teeth numbered 1-8 from the midline outward:
      1 = central incisor (closest to midline)
      8 = third molar / wisdom tooth (furthest from midline)

    The FDI number is: quadrant * 10 + tooth_position
    e.g., Q3 tooth 4 = FDI 34 (mandibular left first premolar)

    Args:
        x: horizontal pixel coordinate
        quadrant: FDI quadrant number (1-4)
        w: image width

    Returns:
        fdi_number: integer FDI tooth number (11-18, 21-28, 31-38, 41-48)
    """
    midline = w / 2

    # Distance from midline (normalized to 0-1)
    if quadrant in [1, 4]:
        # Patient right = image left half, midline is at w/2
        # Closer to midline = higher x value (approaching w/2 from left)
        distance_from_midline = (midline - x) / midline
    else:
        # Patient left = image right half
        # Closer to midline = lower x value (approaching w/2 from right)
        distance_from_midline = (x - midline) / midline

    # Clamp to [0, 1]
    distance_from_midline = max(0.0, min(1.0, distance_from_midline))

    # Map to tooth position 1-8 (1 = closest to midline, 8 = furthest)
    # Invert: distance 0 (at midline) = tooth 1, distance 1 (at edge) = tooth 8
    tooth_position = min(8, max(1, int(distance_from_midline * 8) + 1))

    fdi_number = quadrant * 10 + tooth_position
    return fdi_number


def get_fdi_tooth_name(fdi_number):
    """Return the clinical name for an FDI tooth number."""
    tooth_names = {
        1: "Central Incisor",
        2: "Lateral Incisor",
        3: "Canine",
        4: "First Premolar",
        5: "Second Premolar",
        6: "First Molar",
        7: "Second Molar",
        8: "Third Molar (Wisdom)",
    }
    quadrant_names = {
        1: "Maxillary Right",
        2: "Maxillary Left",
        3: "Mandibular Left",
        4: "Mandibular Right",
    }
    quad = fdi_number // 10
    tooth = fdi_number % 10
    quad_name = quadrant_names.get(quad, f"Q{quad}")
    tooth_name = tooth_names.get(tooth, f"Tooth {tooth}")
    return f"{quad_name} {tooth_name}"


# ==================================================
# 10. Confidence Interpretation & Severity Helpers
# ==================================================
def interpret_confidence(conf):
    """Map a model confidence score to clinical certainty language.

    Args:
        conf: float in [0, 1] (softmax probability)

    Returns:
        str: clinical certainty descriptor
    """
    if conf > 0.85:
        return "confirmed"
    elif conf > 0.65:
        return "likely"
    else:
        return "suspected"


def map_severity(conf):
    """Map a model confidence score to a severity tier.

    Args:
        conf: float in [0, 1] (softmax probability)

    Returns:
        str: severity level
    """
    if conf > 0.85:
        return "high"
    elif conf > 0.7:
        return "moderate"
    else:
        return "low"


# ==================================================
# 11. Confidence Filtering
# ==================================================
def filter_findings(probabilities, heatmaps=None, model_threshold=None, cam_threshold=None, top_k=None):
    """Apply top-K + per-class threshold filtering.

    Only the top_k most confident classes are considered, and each
    must exceed its per-class threshold to be kept.

    Args:
        probabilities: (num_classes,) numpy array of softmax probabilities
        heatmaps: optional dict mapping class_idx -> (H, W) heatmap
        model_threshold: fallback minimum confidence (default from CONFIG)
        cam_threshold: minimum GradCAM peak intensity (default from CONFIG)
        top_k: max number of classes to consider (default from CONFIG)

    Returns:
        valid_classes: list of class indices that passed filtering
    """
    if model_threshold is None:
        model_threshold = CONFIG["model_conf_threshold"]
    if cam_threshold is None:
        cam_threshold = CONFIG["cam_intensity_threshold"]
    if top_k is None:
        top_k = CONFIG.get("top_k", 2)

    class_thresholds = np.array(CONFIG.get("class_thresholds", [model_threshold] * len(probabilities)))

    # Top-K: only consider the top_k most confident classes
    top_indices = np.argsort(probabilities)[-top_k:]

    valid_classes = []
    for cls_idx in top_indices:
        # Per-class threshold
        if probabilities[cls_idx] < class_thresholds[cls_idx]:
            continue
        # GradCAM intensity check (if heatmaps provided)
        if heatmaps and cls_idx in heatmaps:
            if heatmaps[cls_idx].max() < cam_threshold:
                continue
        valid_classes.append(int(cls_idx))

    return valid_classes


# ==================================================
# 11b. WBF Ensemble Functions (DENTEX 2023 Winner)
# ==================================================

def merge_detections_wbf(dino_boxes, dino_scores, dino_labels,
                          yolo_boxes, yolo_scores, yolo_labels,
                          img_w, img_h):
    """Merge disease detections from DINO and YOLO using Weighted Box Fusion.

    Boxes from both models are normalized to [0,1], fused via WBF,
    then denormalized back to pixel coordinates.

    Args:
        dino_boxes:  list of [x,y,w,h] disease boxes from DINO-Swin detector
        dino_scores: list of float confidence scores
        dino_labels: list of int disease class indices
        yolo_boxes:  list of [x,y,w,h] disease boxes from YOLO disease detector
        yolo_scores: list of float confidence scores
        yolo_labels: list of int disease class indices
        img_w, img_h: full image dimensions in pixels

    Returns:
        merged_boxes:  list of [x,y,w,h] in pixels
        merged_scores: list of float merged confidence scores
        merged_labels: list of int merged disease class indices
    """
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ensemble-boxes", "-q"])
        from ensemble_boxes import weighted_boxes_fusion

    # Handle empty inputs
    if not dino_boxes and not yolo_boxes:
        return [], [], []

    def xywh_to_norm_xyxy(boxes, w, h):
        """Convert [x,y,w,h] pixel boxes to [x1,y1,x2,y2] normalized."""
        result = []
        for x, y, bw, bh in boxes:
            x1 = max(0.0, x / w)
            y1 = max(0.0, y / h)
            x2 = min(1.0, (x + bw) / w)
            y2 = min(1.0, (y + bh) / h)
            result.append([x1, y1, x2, y2])
        return result

    boxes_list, scores_list, labels_list = [], [], []

    if dino_boxes:
        boxes_list.append(xywh_to_norm_xyxy(dino_boxes, img_w, img_h))
        scores_list.append(list(dino_scores))
        labels_list.append([float(l) for l in dino_labels])

    if yolo_boxes:
        boxes_list.append(xywh_to_norm_xyxy(yolo_boxes, img_w, img_h))
        scores_list.append(list(yolo_scores))
        labels_list.append([float(l) for l in yolo_labels])

    weights = CONFIG["wbf_weights"][:len(boxes_list)]

    merged_boxes_norm, merged_scores, merged_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=weights,
        iou_thr=CONFIG["wbf_iou_threshold"],
        skip_box_thr=CONFIG["wbf_skip_box_thr"],
        conf_type=CONFIG.get("wbf_conf_type", "max"),
    )

    # Denormalize back to pixel xywh
    merged_boxes_px = []
    for b in merged_boxes_norm:
        x1, y1, x2, y2 = b
        merged_boxes_px.append([
            x1 * img_w, y1 * img_h,
            (x2 - x1) * img_w, (y2 - y1) * img_h,
        ])

    return merged_boxes_px, merged_scores.tolist(), merged_labels.astype(int).tolist()


def compute_iou_xywh(box1, box2):
    """Compute IoU between two [x,y,w,h] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_disease_to_tooth(disease_boxes, tooth_boxes, tooth_fdi_ids):
    """Match each disease bbox to a tooth bbox using IoU (winning team approach).

    Replaces the x-coordinate heuristic with direct spatial IoU overlap.
    Each disease box is matched to the tooth box with highest IoU.

    Args:
        disease_boxes: list of [x,y,w,h] disease detections (WBF output)
        tooth_boxes:   list of [x,y,w,h] tooth detections (tooth YOLO output)
        tooth_fdi_ids: list of FDI numbers for each tooth box

    Returns:
        list of dicts with disease_box, fdi_number, match_iou
    """
    min_iou = CONFIG["tooth_match_iou_threshold"]
    matched = []
    for d_box in disease_boxes:
        best_iou = 0.0
        best_fdi = None
        best_tooth_box = None
        for t_box, fdi_id in zip(tooth_boxes, tooth_fdi_ids):
            iou = compute_iou_xywh(d_box, t_box)
            if iou > best_iou:
                best_iou = iou
                best_fdi = fdi_id
                best_tooth_box = t_box

        # If no IoU match, fall back to x-coordinate heuristic
        if best_fdi is None or best_iou < min_iou:
            cx = d_box[0] + d_box[2] / 2
            cy = d_box[1] + d_box[3] / 2
            # Use img_size as reference (display-scale boxes)
            img_sz = CONFIG["img_size"]
            quadrant = map_to_quadrant(cx, cy, w=img_sz, h=img_sz)
            best_fdi = extract_fdi_tooth(cx, quadrant, w=img_sz)
            best_iou = 0.0

        matched.append({
            "disease_box": d_box,
            "fdi_number": best_fdi,
            "match_iou": best_iou,
            "tooth_box": best_tooth_box,
        })
    return matched


def weighted_vote_fdi(candidate_fdis, candidate_weights):
    """Weighted voting to determine final FDI tooth number.

    Used when multiple overlapping disease boxes may match multiple teeth.
    Confidence scores are used as weights.

    Args:
        candidate_fdis:    list of FDI numbers from candidate matches
        candidate_weights: list of confidence scores for each candidate

    Returns:
        int: winning FDI tooth number
    """
    vote_dict = {}
    for fdi, w in zip(candidate_fdis, candidate_weights):
        vote_dict[fdi] = vote_dict.get(fdi, 0.0) + float(w)
    return max(vote_dict, key=vote_dict.get)


# ==================================================
# 11. Prompt Builder
# ==================================================
def build_prompt(findings):
    """Build a structured clinical prompt from detected findings.

    Uses confidence interpretation to add clinical certainty language
    (confirmed / likely / suspected) and severity tiers, then appends
    explicit LLM instructions to constrain hallucination.

    Args:
        findings: list of dicts, each with keys:
            - disease: str (disease name)
            - fdi_number: int (FDI tooth number)
            - fdi_name: str (clinical tooth name)
            - confidence: float (model probability)
            - certainty: str (confirmed / likely / suspected)
            - severity: str (high / moderate / low)
            - cam_intensity: float (GradCAM peak)
            - quadrant: int (FDI quadrant)
            - location_px: tuple (cx, cy) pixel location

    Returns:
        prompt: str - structured prompt for the LLM
    """
    if not findings:
        return (
            "Patient Radiograph Findings:\n"
            "- No significant pathological findings detected.\n\n"
            "You are a dental radiology assistant.\n\n"
            "Generate a clinical report noting the absence of significant pathology.\n"
            "Format:\n"
            "1. Summary\n"
            "2. Clinical Impression\n"
            "3. Recommendations\n"
        )

    # Sort findings by quadrant then tooth number for clinical readability
    findings_sorted = sorted(findings, key=lambda f: f["fdi_number"])

    # Separate positive and uncertain findings for prompt
    positive_findings = [f for f in findings_sorted if f.get("status") == "positive"]
    uncertain_findings = [f for f in findings_sorted if f.get("status") == "uncertain"]

    lines = []
    if positive_findings:
        lines.append("Confirmed Findings:")
        for f in positive_findings:
            certainty = f.get("certainty", interpret_confidence(f["confidence"]))
            severity = f.get("severity", map_severity(f["confidence"]))
            line = (
                f"- {f['disease']} ({certainty}, severity: {severity}) "
                f"at tooth {f['fdi_number']} ({f['fdi_name']}) "
                f"with {f['confidence']:.0%} confidence"
            )
            lines.append(line)

    if uncertain_findings:
        lines.append("\nUncertain Findings (require clinical verification):")
        for f in uncertain_findings:
            line = (
                f"- [UNCERTAIN] {f['disease']} at tooth {f['fdi_number']} "
                f"({f['fdi_name']}) with {f['confidence']:.0%} confidence "
                f"- borderline detection, recommend follow-up"
            )
            lines.append(line)

    # Build finding summary by quadrant
    quadrant_summary = {}
    for f in findings_sorted:
        q = f["quadrant"]
        if q not in quadrant_summary:
            quadrant_summary[q] = []
        quadrant_summary[q].append(f)

    # Construct the full prompt
    prompt = "Patient Radiograph Findings:\n"
    prompt += "\n".join(lines)
    prompt += "\n\n"

    # Add quadrant context
    quadrant_names = {1: "upper right", 2: "upper left", 3: "lower left", 4: "lower right"}
    affected_quads = [quadrant_names[q] for q in sorted(quadrant_summary.keys())]
    prompt += f"Affected regions: {', '.join(affected_quads)}\n"
    prompt += f"Total findings: {len(findings)}\n\n"

    # Structured LLM instructions to reduce hallucination
    prompt += (
        "You are a dental radiology assistant.\n\n"
        "Generate a clinical report using ONLY the findings above.\n\n"
        "Rules:\n"
        "- Do NOT introduce new diseases or teeth not listed above.\n"
        "- Use cautious medical language: 'suggestive of' for suspected, "
        "'consistent with' for likely, 'indicative of' for confirmed.\n"
        "- Respect confidence levels: flag low-severity findings as requiring follow-up.\n\n"
        "Format:\n"
        "1. Summary\n"
        "2. Detailed Findings (tooth-wise)\n"
        "3. Clinical Impression\n"
        "4. Recommendations\n"
    )

    return prompt


# ==================================================
# 12. DentalVisionPipeline
# ==================================================
class DentalVisionPipeline:
    """Unified pipeline: Image Path -> Clinical Prompt + Visualizations.

    DENTEX 2023 Winner Approach:
      - WBF merges DINO-Swin + YOLO disease detections (authoritative findings)
      - IoU-based tooth matching replaces x-coordinate heuristic
      - GradCAM kept as visualization overlay only (does NOT gate predictions)
      - Falls back to two-stage crop classifier if WBF models unavailable

    Usage:
        pipeline = DentalVisionPipeline(model, config)
        result = pipeline.process("path/to/xray.png")
        print(result["prompt"])
        pipeline.visualize(result)
    """
    def __init__(self, model, config, gradcam_layer="conv_head"):
        self.model = model
        self.config = config
        self.disease_names = config["disease_names"]
        self.gradcam = GradCAM(model, target_layer_name=gradcam_layer)
        self.yolo_model = None
        self.yolo_disease_model = None

        # Load YOLO tooth detection model (for IoU tooth matching in WBF mode)
        yolo_path = None
        for candidate in config.get("yolo_model_paths", []):
            if os.path.exists(candidate):
                yolo_path = candidate
                break

        if yolo_path:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(yolo_path)
                print(f"YOLO tooth detector loaded: {yolo_path}")
            except Exception as e:
                print(f"WARNING: Failed to load YOLO tooth model: {e}")
                self.yolo_model = None
        else:
            print("WARNING: No YOLO tooth model found.")
            for p in config.get("yolo_model_paths", []):
                print(f"  Searched: {p}")

        # Load YOLO disease detector (2nd WBF ensemble member from Phase 0.5 Part 2)
        yolo_disease_path = None
        for candidate in config.get("yolo_disease_model_paths", []):
            if os.path.exists(candidate):
                yolo_disease_path = candidate
                break

        if yolo_disease_path:
            try:
                from ultralytics import YOLO as _YOLO
                self.yolo_disease_model = _YOLO(yolo_disease_path)
                print(f"YOLO disease detector loaded: {yolo_disease_path}")
            except Exception as e:
                print(f"WARNING: Failed to load YOLO disease model: {e}")
                self.yolo_disease_model = None
        else:
            print("WARNING: No YOLO disease model found. WBF will run on single model.")
            for p in config.get("yolo_disease_model_paths", []):
                print(f"  Searched: {p}")
                print("  Add this to your Kaggle notebook before running Phase 2:")
                print("    !pip install ultralytics -q")


    def _run_yolo_disease_detections(self, img_path, img_w=None, img_h=None):
        """Run YOLO disease detector on full image, return display-scale xywh boxes."""
        if self.yolo_disease_model is None:
            return [], [], []
        results = self.yolo_disease_model(
            img_path,
            conf=self.config.get("yolo_disease_conf_threshold", 0.3),
            verbose=False,
        )[0]
        boxes, scores, labels = [], [], []
        if results.boxes is not None and len(results.boxes) > 0:
            disp_size = self.config["img_size"]
            scale_x = (disp_size / img_w) if img_w else 1.0
            scale_y = (disp_size / img_h) if img_h else 1.0
            # Map YOLO class idx -> Phase 1 disease class idx
            # YOLO disease classes: sorted [Caries, Deep Caries, Impacted, Periapical]
            yolo_to_phase1 = {0: 0, 1: 1, 2: 2, 3: 3}
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Keep YOLO disease boxes in the same 336x336 display space as
                # tooth-crop boxes before sending them into WBF and IoU matching.
                dx1, dy1 = float(x1 * scale_x), float(y1 * scale_y)
                dx2, dy2 = float(x2 * scale_x), float(y2 * scale_y)
                boxes.append([dx1, dy1, dx2 - dx1, dy2 - dy1])
                scores.append(float(box.conf[0].cpu()))
                yolo_cls = int(box.cls[0].cpu())
                labels.append(yolo_to_phase1.get(yolo_cls, yolo_cls))
        return boxes, scores, labels

    def _get_tooth_boxes_with_fdi(self, img_path, img_w, img_h):
        """Run tooth YOLO and assign FDI numbers to each detected tooth box.

        Returns:
            tooth_boxes:  list of [x,y,w,h] in pixels
            tooth_fdi_ids: list of int FDI tooth numbers
        """
        if self.yolo_model is None:
            return [], []
        results = self.yolo_model(
            img_path,
            conf=self.config.get("yolo_conf_threshold", 0.45),
            iou=self.config.get("yolo_iou_threshold", 0.6),
            verbose=False,
        )[0]
        tooth_boxes, tooth_fdi_ids = [], []
        if results.boxes is not None and len(results.boxes) > 0:
            disp_w = self.config["img_size"]
            disp_h = self.config["img_size"]
            scale_x = disp_w / img_w
            scale_y = disp_h / img_h
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Scale to display size (same space as disease boxes)
                dx1, dy1 = x1 * scale_x, y1 * scale_y
                dx2, dy2 = x2 * scale_x, y2 * scale_y
                dw, dh = dx2 - dx1, dy2 - dy1
                tooth_boxes.append([dx1, dy1, dw, dh])
                # Assign FDI via x-coordinate heuristic for tooth identity
                cx = (dx1 + dx2) / 2
                cy = (dy1 + dy2) / 2
                quadrant = map_to_quadrant(cx, cy, w=disp_w, h=disp_h)
                fdi = extract_fdi_tooth(cx, quadrant, w=disp_w)
                tooth_fdi_ids.append(fdi)
        return tooth_boxes, tooth_fdi_ids

    def _run_wbf_ensemble(self, img_path, image_rgb):
        """Run the DENTEX 2023 winning WBF ensemble pipeline.

        Step 1: YOLO disease detector -> disease boxes (ensemble member 2)
        Step 2: Phase 1 crop classifier on YOLO-tooth crops -> class probabilities (member 1)
        Step 3: WBF merges both detector outputs
        Step 4: IoU-based tooth matching assigns FDI to each disease box
        Step 5: GradCAM attached as visualization overlay (not a gate)

        Returns list of finding dicts.
        """
        img_h, img_w = image_rgb.shape[:2]
        disp_size = self.config["img_size"]
        display_img = cv2.resize(image_rgb, (disp_size, disp_size))
        scale_x = disp_size / img_w
        scale_y = disp_size / img_h

        print("  WBF ensemble mode (DENTEX 2023 winner approach)")

        # --- Member 2: YOLO disease detector ---
        yolo_boxes, yolo_scores, yolo_labels = self._run_yolo_disease_detections(
            img_path, img_w=img_w, img_h=img_h
        )
        print(f"  YOLO disease detector: {len(yolo_boxes)} detections")

        # --- Member 1: Phase 1 crop classifier on tooth crops ---
        # Run tooth YOLO to get crop regions, then classify each crop
        dino_boxes, dino_scores, dino_labels = [], [], []
        tooth_boxes_raw, _ = self._get_tooth_boxes_with_fdi(img_path, img_w, img_h)

        if tooth_boxes_raw:
            val_transform_local = A.Compose([
                A.Resize(self.config["img_size"], self.config["img_size"]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
            class_thresholds = np.array(
                self.config.get("class_thresholds", [self.config["wbf_skip_box_thr"]] * self.config["num_classes"])
            )
            tooth_crop_min_conf = float(self.config.get("tooth_crop_min_conf_threshold", 0.60))
            for t_box in tooth_boxes_raw:
                x, y, w, h = t_box
                # Extract crop from original image (non-scaled)
                orig_x = x / scale_x
                orig_y = y / scale_y
                orig_w = w / scale_x
                orig_h = h / scale_y
                pad = 0.15
                x1 = max(0, int(orig_x - orig_w * pad))
                y1 = max(0, int(orig_y - orig_h * pad))
                x2 = min(img_w, int(orig_x + orig_w + orig_w * pad))
                y2 = min(img_h, int(orig_y + orig_h + orig_h * pad))
                crop = image_rgb[y1:y2, x1:x2]
                if crop.shape[0] < 5 or crop.shape[1] < 5:
                    continue
                aug = val_transform_local(image=crop)
                crop_tensor = aug["image"].unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = self.model(crop_tensor)
                    if self.config.get("tta_enabled", True):
                        logits = (logits + self.model(torch.flip(crop_tensor, dims=[3]))) / 2.0
                    logits = logits / self.config.get("temperature", 1.5)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                pred_class = int(np.argmax(probs))
                pred_conf  = float(probs[pred_class])

                # Per-class threshold (same logic as WBF filtering)
                crop_uncertain_margin = float(self.config.get("uncertain_margin", 0.1))
                if pred_class < len(class_thresholds):
                    crop_threshold = float(class_thresholds[pred_class])
                else:
                    crop_threshold = tooth_crop_min_conf

                # Apply class-specific adjustments
                crop_adjustments = self.config.get("class_threshold_adjustments", {})
                crop_disease_name = self.config["disease_names"].get(pred_class, "")
                crop_threshold += crop_adjustments.get(crop_disease_name, 0.0)

                # 3-tier: accept / pass as borderline / skip
                if pred_conf >= crop_threshold:
                    pass  # Accept normally
                elif pred_conf >= crop_threshold - crop_uncertain_margin:
                    print(f"    [BORDERLINE] crop passed (conf={pred_conf:.2f}, threshold={crop_threshold:.2f}, margin={crop_uncertain_margin}) | {crop_disease_name}")
                else:
                    print(f"    [LOW-CONF] crop skipped (conf={pred_conf:.2f} < {crop_threshold - crop_uncertain_margin:.2f}) | all=[{', '.join(f'{p:.2f}' for p in probs)}]")
                    continue

                # Scale box to display coords for WBF
                dino_boxes.append([x, y, w, h])
                dino_scores.append(pred_conf)
                dino_labels.append(pred_class)

        print(f"  Phase 1 classifier (tooth crops): {len(dino_boxes)} disease detections")

        # --- WBF: merge both detector outputs ---
        # Disease boxes are in display-scale (disp_size x disp_size)
        merged_boxes, merged_scores, merged_labels = merge_detections_wbf(
            dino_boxes, dino_scores, dino_labels,
            yolo_boxes, yolo_scores, yolo_labels,
            img_w=disp_size, img_h=disp_size,
        )
        print(f"  WBF merged: {len(merged_boxes)} final disease detections")

        if not merged_boxes:
            return [], display_img, {}

        # --- IoU tooth matching ---
        tooth_boxes_disp, tooth_fdi_ids = self._get_tooth_boxes_with_fdi(img_path, img_w, img_h)
        matched = match_disease_to_tooth(merged_boxes, tooth_boxes_disp, tooth_fdi_ids)

        # --- Build findings from matched detections ---
        findings = []
        composite_heatmaps = {}
        final_class_thresholds = np.array(
            self.config.get("class_thresholds", [self.config["model_conf_threshold"]] * self.config["num_classes"])
        )
        final_min_conf = float(self.config.get("wbf_final_min_conf_threshold", 0.60))
        skipped_low_wbf = 0

        caries_suppression = float(self.config.get("caries_suppression_threshold", 0.80))
        uncertain_margin = float(self.config.get("uncertain_margin", 0.08))
        threshold_adjustments = self.config.get("class_threshold_adjustments", {})

        for m, score, label in zip(matched, merged_scores, merged_labels):
            label = int(label)
            score = float(score)
            disease_name_check = self.disease_names.get(label, f"Class {label}")

           
            # wbf_final_min_conf is only used as fallback when no per-class threshold exists
            if label < len(final_class_thresholds):
                report_threshold = float(final_class_thresholds[label])
            else:
                report_threshold = final_min_conf

            # Class-specific adjustment (e.g., Periapical -0.05, Caries +0.05)
            adjustment = threshold_adjustments.get(disease_name_check, 0.0)
            report_threshold += adjustment

            # Caries suppression (additional safety net)
            if disease_name_check == "Caries":
                report_threshold = max(report_threshold, caries_suppression)

            # 3-tier decision: positive / uncertain / skip
            if score >= report_threshold:
                detection_status = "positive"
            elif score >= report_threshold - uncertain_margin:
                detection_status = "uncertain"
            else:
                skipped_low_wbf += 1
                print(
                    f"    [SKIP] {disease_name_check}: "
                    f"conf={score:.2f} < {report_threshold - uncertain_margin:.2f} "
                    f"(threshold={report_threshold:.2f}, margin={uncertain_margin})"
                )
                continue

            disease_box = m["disease_box"]
            fdi_number = m["fdi_number"]
            fdi_name = get_fdi_tooth_name(fdi_number)
            x, y, w, h = disease_box
            cx = x + w / 2
            cy = y + h / 2
            quadrant = map_to_quadrant(cx, cy, w=disp_size, h=disp_size)
            disease_name = self.disease_names.get(label, f"Class {label}")

            # GradCAM as visualization overlay (not a gate)
            cam_intensity = 0.0
            if self.config.get("generate_gradcam_overlays", True):
                try:
                    # Extract crop for GradCAM from display image
                    x1c = max(0, int(x))
                    y1c = max(0, int(y))
                    x2c = min(disp_size, int(x + w))
                    y2c = min(disp_size, int(y + h))
                    crop_disp = display_img[y1c:y2c, x1c:x2c]
                    if crop_disp.shape[0] >= 5 and crop_disp.shape[1] >= 5:
                        aug_local = A.Compose([
                            A.Resize(self.config["img_size"], self.config["img_size"]),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensorV2(),
                        ])
                        crop_tensor = aug_local(image=crop_disp)["image"].unsqueeze(0).to(device)
                        cam = self.gradcam.generate(crop_tensor, label)
                        cam_intensity = self.gradcam.compute_cam_intensity(cam)
                        # Composite back onto full image canvas
                        cw = max(1, x2c - x1c)
                        ch = max(1, y2c - y1c)
                        cam_resized = cv2.resize(cam, (cw, ch))
                        if label not in composite_heatmaps:
                            composite_heatmaps[label] = np.zeros((disp_size, disp_size), dtype=np.float32)
                        region = composite_heatmaps[label][y1c:y1c+ch, x1c:x1c+cw]
                        if region.shape == cam_resized.shape:
                            composite_heatmaps[label][y1c:y1c+ch, x1c:x1c+cw] = np.maximum(region, cam_resized)
                except Exception:
                    pass  # GradCAM is optional, never blocks findings

            finding = {
                "disease": disease_name,
                "disease_idx": label,
                "fdi_number": fdi_number,
                "fdi_name": fdi_name,
                "quadrant": quadrant,
                "confidence": score,
                "certainty": interpret_confidence(score),
                "severity": map_severity(score),
                "cam_intensity": cam_intensity,
                "match_iou": m["match_iou"],
                "status": detection_status,
                "location_px": (int(cx), int(cy)),
                "bbox": (int(x), int(y), int(w), int(h)),
            }
            findings.append(finding)

        # Per-tooth limit: keep top N findings per tooth (prevents multi-disease spam)
        max_per_tooth = int(self.config.get("max_findings_per_tooth", 2))
        from collections import defaultdict
        tooth_groups = defaultdict(list)
        for f in findings:
            tooth_groups[f["fdi_number"]].append(f)
        findings = []
        for fdi, group in tooth_groups.items():
            group_sorted = sorted(group, key=lambda f: f["confidence"], reverse=True)
            findings.extend(group_sorted[:max_per_tooth])

        # Cap at max_findings
        max_f = self.config.get("max_findings", 12)
        if len(findings) > max_f:
            findings = sorted(findings, key=lambda f: f["confidence"], reverse=True)[:max_f]

        if skipped_low_wbf:
            print(f"  WBF report gate skipped {skipped_low_wbf} low-confidence merged detections")

        positive_count = sum(1 for f in findings if f["status"] == "positive")
        uncertain_count = sum(1 for f in findings if f["status"] == "uncertain")
        print(f"  Final findings: {len(findings)} ({positive_count} positive, {uncertain_count} uncertain)")
        for f in findings:
            marker = "[+]" if f["status"] == "positive" else "[?]"
            print(f"    {marker} [{f['disease']}] Tooth {f['fdi_number']} "
                  f"conf={f['confidence']:.2f} IoU={f['match_iou']:.2f} [{f['certainty']}] "
                  f"status={f['status']}")

        heatmaps = {k: v for k, v in composite_heatmaps.items() if v.max() > 0.01}
        return findings, display_img, heatmaps

    def process(self, img_path, gt_labels=None, gt_teeth=None):
        """Run the full pipeline on a single dental X-ray.

        If use_wbf_ensemble=True (default): runs DENTEX 2023 winner WBF pipeline.
        Falls back to two-stage crop classifier if WBF models are unavailable.

        Args:
            img_path:  path to the dental panoramic X-ray image
            gt_labels: optional image-level GT labels for display
            gt_teeth:  optional list of {bbox, labels, disease_name} dicts

        Returns:
            dict with prompt, findings, probabilities, display_img, etc.
        """
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        print(f"\nImage: {os.path.basename(img_path)}")

        # -----------------------------------------------
        # PRIMARY PATH: WBF Ensemble (DENTEX 2023 winner)
        # -----------------------------------------------
        if self.config.get("use_wbf_ensemble", True):
            findings, display_img, heatmaps = self._run_wbf_ensemble(img_path, image_rgb)
            prompt = build_prompt(findings)
            probabilities = np.zeros(self.config["num_classes"])
            for f in findings:
                probabilities[f["disease_idx"]] = max(probabilities[f["disease_idx"]], f["confidence"])
            return {
                "prompt": prompt,
                "findings": findings,
                "probabilities": probabilities,
                "heatmaps": heatmaps,
                "display_img": display_img,
                "img_path": img_path,
                "gt_labels": gt_labels,
                "gt_teeth": load_gt_boxes_for_image(img_path, val_teeth_data, self.disease_names),
                "orig_shape": image_rgb.shape[:2],
                "mode": "wbf_ensemble",
            }

        # -----------------------------------------------
        # FALLBACK PATH: Two-stage crop classifier
        # -----------------------------------------------
        print("  Fallback: two-stage crop classifier mode")
        display_img = cv2.resize(image_rgb, (self.config["img_size"], self.config["img_size"]))

        # --- Resolve tooth bounding boxes ---
        # If use_yolo_for_eval=True, YOLO is preferred even when GT is available
        use_yolo = self.config.get("use_yolo_for_eval", False)

        if use_yolo and self.yolo_model is not None:
            # YOLO-first mode: use predicted detections for realistic eval
            yolo_results = self.yolo_model(
                img_path,
                conf=self.config.get("yolo_conf_threshold", 0.25),
                iou=self.config.get("yolo_iou_threshold", 0.6),
                verbose=False,
            )[0]
            detected_teeth = []
            if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
                for box in yolo_results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().item()
                    detected_teeth.append({
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "yolo_conf": conf,
                    })
            teeth_source = "yolo"
            print(f"Two-stage mode: {len(detected_teeth)} tooth crops (source: YOLO, conf>={self.config.get('yolo_conf_threshold', 0.25)})")
        elif gt_teeth and len(gt_teeth) > 0 and not use_yolo:
            # GT mode: use ground-truth boxes (ideal conditions)
            teeth_source = "gt"
            detected_teeth = gt_teeth
            print(f"Two-stage mode: {len(detected_teeth)} tooth crops (source: GT manifest)")
        elif gt_teeth and len(gt_teeth) > 0 and use_yolo and self.yolo_model is None:
            # Fallback: YOLO requested but not available, use GT with warning
            teeth_source = "gt_fallback"
            detected_teeth = gt_teeth
            print(f"Two-stage mode: {len(detected_teeth)} tooth crops (source: GT fallback - YOLO unavailable)")
        elif self.yolo_model is not None:
            # Standard YOLO inference (no GT available)
            yolo_results = self.yolo_model(
                img_path,
                conf=self.config.get("yolo_conf_threshold", 0.25),
                iou=self.config.get("yolo_iou_threshold", 0.6),
                verbose=False,
            )[0]
            detected_teeth = []
            if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
                for box in yolo_results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().item()
                    detected_teeth.append({
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "yolo_conf": conf,
                    })
            teeth_source = "yolo"
            print(f"Two-stage mode: {len(detected_teeth)} tooth crops (source: YOLO, conf>={self.config.get('yolo_conf_threshold', 0.25)})")
        else:
            detected_teeth = []

        # --- Two-stage: crop each tooth and classify ---
        if detected_teeth and len(detected_teeth) > 0:
            all_crop_probs = []
            findings = []
            img_h, img_w = image_rgb.shape[:2]
            scale_x = self.config["img_size"] / img_w
            scale_y = self.config["img_size"] / img_h
            disp_size = self.config["img_size"]

            # Accumulate per-class heatmaps on the display-size canvas
            composite_heatmaps = {}
            for cls_idx in range(self.config["num_classes"]):
                composite_heatmaps[cls_idx] = np.zeros((disp_size, disp_size), dtype=np.float32)

            for tooth_info in detected_teeth:
                bbox = tooth_info["bbox"]
                x, y, w, h = bbox

                # Extract padded crop (same as Phase 1 training)
                pad_ratio = 0.15
                pad_w = int(w * pad_ratio)
                pad_h = int(h * pad_ratio)
                x1 = max(0, int(x) - pad_w)
                y1 = max(0, int(y) - pad_h)
                x2 = min(img_w, int(x + w) + pad_w)
                y2 = min(img_h, int(y + h) + pad_h)
                crop = image_rgb[y1:y2, x1:x2]

                if crop.shape[0] < 5 or crop.shape[1] < 5:
                    continue

                # Preprocess crop for model
                augmented = val_transform(image=crop)
                crop_tensor = augmented["image"].unsqueeze(0).to(device)

                # Classify this crop (softmax single-label + TTA)
                with torch.no_grad():
                    logits = self.model(crop_tensor)
                    # TTA: horizontal flip augmentation
                    if self.config.get("tta_enabled", True):
                        logits_flip = self.model(torch.flip(crop_tensor, dims=[3]))
                        logits = (logits + logits_flip) / 2.0
                    # Temperature scaling: smooths overconfident softmax for better ranking
                    temp = self.config.get("temperature", 1.5)
                    logits = logits / temp
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

                # TOP-1 per crop (single-label classifier)
                pred_class = int(np.argmax(probs))
                pred_conf = float(probs[pred_class])

                # Skip low-confidence disease predictions so every detected
                # tooth is not automatically turned into a finding.
                class_thresholds = np.array(self.config.get("class_thresholds", [0.5] * self.config["num_classes"]))
                pred_threshold = class_thresholds[pred_class] if pred_class < len(class_thresholds) else self.config["model_conf_threshold"]
                pred_threshold = max(float(pred_threshold), float(self.config.get("tooth_crop_min_conf_threshold", 0.60)))
                if pred_conf < pred_threshold:
                    print(f"  Crop ({int(x)},{int(y)},{int(w)},{int(h)}) -> "
                          f"LOW-CONF {self.disease_names[pred_class]} "
                          f"({pred_conf:.3f} < {pred_threshold:.2f}) "
                          f"| [{', '.join(f'{p:.2f}' for p in probs)}]")
                    continue

               
        
                sorted_probs = np.sort(probs)[::-1]
                margin = sorted_probs[0] - sorted_probs[1]
                margin_thresh = self.config.get("margin_threshold", 0.15)
                if margin < margin_thresh:
                    print(f"  Crop ({int(x)},{int(y)},{int(w)},{int(h)}) -> "
                          f"AMBIGUOUS {self.disease_names[pred_class]} "
                          f"(margin={margin:.3f} < {margin_thresh}) "
                          f"| [{', '.join(f'{p:.2f}' for p in probs)}]")
                    continue

                all_crop_probs.append(probs)

                print(f"  Crop ({int(x)},{int(y)},{int(w)},{int(h)}) -> "
                      f"{self.disease_names[pred_class]} ({pred_conf:.3f}) "
                      f"| [{', '.join(f'{p:.2f}' for p in probs)}]")

                # Generate GradCAM only for predicted class
                crop_heatmaps = self.gradcam.generate_all_classes(crop_tensor, probs)

                # Map crop GradCAM back to full-image display coordinates
                disp_x1 = int(x1 * scale_x)
                disp_y1 = int(y1 * scale_y)
                disp_x2 = int(x2 * scale_x)
                disp_y2 = int(y2 * scale_y)
                disp_cw = max(1, disp_x2 - disp_x1)
                disp_ch = max(1, disp_y2 - disp_y1)

                for cls_idx, cam in crop_heatmaps.items():
                    cam_resized = cv2.resize(cam, (disp_cw, disp_ch))
                    region = composite_heatmaps[cls_idx][disp_y1:disp_y1+disp_ch, disp_x1:disp_x1+disp_cw]
                    if region.shape == cam_resized.shape:
                        composite_heatmaps[cls_idx][disp_y1:disp_y1+disp_ch, disp_x1:disp_x1+disp_cw] = \
                            np.maximum(region, cam_resized)

                # --- Three-tier confidence zones ---
                uncertain_thresh = self.config.get("uncertain_threshold", 0.35)
                positive_thresh = self.config.get("model_conf_threshold", 0.5)

                # Below uncertain threshold = ignore entirely
                if pred_conf < uncertain_thresh:
                    continue

                # Build finding directly from this crop's TOP-1 prediction
                # Use crop center mapped to display coordinates for FDI
                crop_cx = int((disp_x1 + disp_x2) / 2)
                crop_cy = int((disp_y1 + disp_y2) / 2)
                quadrant = map_to_quadrant(crop_cx, crop_cy, w=disp_size, h=disp_size)
                fdi_number = extract_fdi_tooth(crop_cx, quadrant, w=disp_size)
                fdi_name = get_fdi_tooth_name(fdi_number)

                # Compute meaningful GradCAM intensity (top-10% mean, not max)
                cam_heatmap = crop_heatmaps.get(pred_class, np.zeros((1, 1)))
                cam_intensity = self.gradcam.compute_cam_intensity(cam_heatmap)

                # Determine certainty tier
                if pred_conf >= positive_thresh:
                    status = "positive"
                else:
                    status = "uncertain"  # Between uncertain_thresh and positive_thresh

                finding = {
                    "disease": self.disease_names[pred_class],
                    "disease_idx": pred_class,
                    "fdi_number": fdi_number,
                    "fdi_name": fdi_name,
                    "quadrant": quadrant,
                    "confidence": pred_conf,
                    "certainty": interpret_confidence(pred_conf),
                    "severity": map_severity(pred_conf),
                    "cam_intensity": cam_intensity,
                    "status": status,
                    "location_px": (crop_cx, crop_cy),
                    "bbox": (disp_x1, disp_y1, disp_cw, disp_ch),
                }
                findings.append(finding)

            # Image-level probability summary (max across all crops)
            if all_crop_probs:
                probabilities = np.max(np.array(all_crop_probs), axis=0)
            else:
                probabilities = np.zeros(self.config["num_classes"])

            # Keep only heatmaps with actual activations
            heatmaps = {k: v for k, v in composite_heatmaps.items() if v.max() > 0.01}

            # Print predictions with threshold tags
            class_thresholds = np.array(self.config.get("class_thresholds", [0.5]*self.config["num_classes"]))
            print("\nModel predictions (softmax, single-label + TTA):")
            for idx, prob in enumerate(probabilities):
                thresh = class_thresholds[idx] if idx < len(class_thresholds) else 0.5
                if prob >= thresh:
                    p_status = "POSITIVE"
                elif prob >= self.config.get("uncertain_threshold", 0.35):
                    p_status = "UNCERTAIN"
                else:
                    p_status = "negative"
                print(f"  {self.disease_names[idx]}: {prob:.4f} [{p_status}] (threshold={thresh:.2f})")

            # Print GT comparison
            if gt_labels is not None:
                gt_disease_names = [self.disease_names[i] for i in range(self.config["num_classes"]) if i < len(gt_labels) and gt_labels[i] > 0.5]
                gt_display = ", ".join(gt_disease_names) if gt_disease_names else "None"
                print(f"\n--- Ground Truth ---")
                print(f"    Diseases: {gt_display}")
                print(f"    Raw labels: {gt_labels}")

            # --- Deduplicate: keep only highest-confidence class per tooth ---
            deduplicated = {}
            for f in findings:
                key = f["fdi_number"]  # One finding per tooth (not per disease+tooth)
                if key not in deduplicated or f["confidence"] > deduplicated[key]["confidence"]:
                    deduplicated[key] = f
            findings = list(deduplicated.values())

            # --- Cap findings at max_findings (keep top-K by confidence) ---
            max_findings = self.config.get("max_findings", 8)
            if len(findings) > max_findings:
                findings = sorted(findings, key=lambda f: f["confidence"], reverse=True)[:max_findings]

            prompt = build_prompt(findings)

            # Count positive vs uncertain
            n_positive = sum(1 for f in findings if f["status"] == "positive")
            n_uncertain = sum(1 for f in findings if f["status"] == "uncertain")
            print(f"\nFindings: {len(findings)} ({n_positive} positive, {n_uncertain} uncertain)")
            for f in findings:
                marker = "[+]" if f["status"] == "positive" else "[?]"
                print(f"  {marker} [{f['disease']}] Tooth {f['fdi_number']} ({f['fdi_name']}) "
                      f"conf={f['confidence']:.2f} (cam={f['cam_intensity']:.2f}) "
                      f"[{f['certainty']}/{f['severity']}]")

            return {
                "prompt": prompt,
                "findings": findings,
                "probabilities": probabilities,
                "heatmaps": heatmaps,
                "display_img": display_img,
                "img_path": img_path,
                "gt_labels": gt_labels,
            }

        # --- Fallback: full-image mode (legacy, for images without bbox info) ---
        print("Fallback: full-image mode (no tooth bboxes available)")
        input_tensor, display_img = preprocess_image(img_path)

        with torch.no_grad():
            logits = self.model(input_tensor)
            # TTA: horizontal flip
            if self.config.get("tta_enabled", True):
                logits_flip = self.model(torch.flip(input_tensor, dims=[3]))
                logits = (logits + logits_flip) / 2.0
            # Temperature scaling
            temp = self.config.get("temperature", 1.5)
            logits = logits / temp
        probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        # Sanity check
        print(f"  Top class: {self.disease_names[np.argmax(probabilities)]} ({probabilities.max():.4f})")

        class_thresholds = np.array(self.config.get("class_thresholds", [0.5]*self.config["num_classes"]))
        print("\nModel predictions (softmax, single-label + TTA):")
        for idx, prob in enumerate(probabilities):
            thresh = class_thresholds[idx] if idx < len(class_thresholds) else 0.5
            if prob >= thresh:
                status = "POSITIVE"
            elif prob >= thresh * 0.7:
                status = "UNCERTAIN"
            else:
                status = "negative"
            gt_str = ""
            if gt_labels is not None:
                gt_val = gt_labels[idx] if idx < len(gt_labels) else 0
                gt_str = f"  | GT: {'YES' if gt_val > 0.5 else 'no'}"
            print(f"  {self.disease_names[idx]}: {prob:.4f} [{status}] (threshold={thresh:.2f}){gt_str}")

        heatmaps = self.gradcam.generate_all_classes(input_tensor, probabilities)
        valid_classes = filter_findings(probabilities, heatmaps)

        if not valid_classes:
            print("No findings passed filter.")
            return {
                "prompt": build_prompt([]),
                "findings": [],
                "probabilities": probabilities,
                "heatmaps": heatmaps,
                "display_img": display_img,
                "img_path": img_path,
                "gt_labels": gt_labels,
            }

        findings = []
        for cls_idx in valid_classes:
            heatmap = heatmaps[cls_idx]
            contours, binary, centers, bboxes = threshold_heatmap(heatmap)
            for center, bbox in zip(centers, bboxes):
                cx, cy = center
                quadrant = map_to_quadrant(cx, cy, w=self.config["img_size"], h=self.config["img_size"])
                fdi_number = extract_fdi_tooth(cx, quadrant, w=self.config["img_size"])
                fdi_name = get_fdi_tooth_name(fdi_number)
                conf = float(probabilities[cls_idx])
                finding = {
                    "disease": self.disease_names[cls_idx],
                    "disease_idx": cls_idx,
                    "fdi_number": fdi_number,
                    "fdi_name": fdi_name,
                    "quadrant": quadrant,
                    "confidence": conf,
                    "certainty": interpret_confidence(conf),
                    "severity": map_severity(conf),
                    "cam_intensity": float(heatmap.max()),
                    "location_px": (cx, cy),
                    "bbox": bbox,
                }
                findings.append(finding)

        deduplicated = {}
        for f in findings:
            key = (f["disease_idx"], f["fdi_number"])
            if key not in deduplicated or f["cam_intensity"] > deduplicated[key]["cam_intensity"]:
                deduplicated[key] = f
        findings = list(deduplicated.values())

        prompt = build_prompt(findings)

        class_thresholds = np.array(self.config.get("class_thresholds", [0.5]*self.config["num_classes"]))
        n_positive = sum(1 for f in findings if f["confidence"] >= class_thresholds[f["disease_idx"]])
        n_uncertain = len(findings) - n_positive
        print(f"\nFindings: {len(findings)} ({n_positive} positive, {n_uncertain} uncertain)")
        for f in findings:
            thresh = class_thresholds[f["disease_idx"]] if f["disease_idx"] < len(class_thresholds) else 0.5
            marker = "[+]" if f["confidence"] >= thresh else "[?]"
            print(f"  {marker} [{f['disease']}] Tooth {f['fdi_number']} ({f['fdi_name']}) "
                  f"conf={f['confidence']:.2f} (raw={f['confidence']:.2f}, cam={f['cam_intensity']:.2f}) "
                  f"[{f['certainty']}/{f['severity']}]")

        return {
            "prompt": prompt,
            "findings": findings,
            "probabilities": probabilities,
            "heatmaps": heatmaps,
            "display_img": display_img,
            "img_path": img_path,
            "gt_labels": gt_labels,
            "gt_teeth": load_gt_boxes_for_image(img_path, val_teeth_data, self.disease_names) if val_teeth_data else [],
            "orig_shape": image_rgb.shape[:2],
        }

    def visualize(self, result, save_path=None):
        """Create a multi-panel visualization with GT vs Predicted comparison.

        Panel layout:
          [Original X-ray] [GradCAM Overlay(s)] [Predicted Findings] [Ground Truth]

        GT panel shows green dashed boxes from DENTEX annotations so you can
        visually validate whether model predictions spatially match the labels.
        """
        display_img   = result["display_img"]
        findings      = result["findings"]
        heatmaps      = result["heatmaps"]
        probabilities = result["probabilities"]
        gt_teeth      = result.get("gt_teeth", [])
        img_size      = self.config["img_size"]
        img_h_orig, img_w_orig = result.get("orig_shape", (img_size, img_size))
        scale_x = img_size / img_w_orig
        scale_y = img_size / img_h_orig

        # Panels: Original | GradCAM(s, max 2) | Predicted Findings | Ground Truth
        detected_classes = list(heatmaps.keys())
        n_cam_panels = min(len(detected_classes), 2)
        n_panels = 1 + n_cam_panels + 1 + (1 if gt_teeth is not None else 0)

        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5.5))
        if n_panels == 1:
            axes = [axes]

        # Panel 0: Original image
        axes[0].imshow(display_img)
        axes[0].set_title("Original X-ray", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Middle panels: GradCAM overlays (up to 2)
        for i, cls_idx in enumerate(detected_classes[:n_cam_panels]):
            ax = axes[i + 1]
            heatmap = heatmaps[cls_idx]
            heatmap_colored = cv2.applyColorMap(
                (heatmap * 255).astype(np.uint8), CONFIG["cam_colormap"]
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(display_img, 0.5, heatmap_colored, 0.5, 0)
            ax.imshow(overlay)
            ax.set_title(
                f"GradCAM: {self.disease_names[cls_idx]}\n(conf={probabilities[cls_idx]:.2f})",
                fontsize=10, fontweight="bold",
            )
            ax.axis("off")

        # Predicted Findings panel (colored boxes with uncertain distinction)
        pred_colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#A8E6CF", "#C9A0DC"]
        uncertain_color = "#FFA500"  # Orange for uncertain findings
        ax_pred = axes[1 + n_cam_panels]
        ax_pred.imshow(display_img)
        ax_pred.set_title("Predicted Findings", fontsize=11, fontweight="bold", color="#CC2200")

        positive_findings = [f for f in findings if f.get("status") == "positive"]
        uncertain_findings = [f for f in findings if f.get("status") == "uncertain"]

        # Draw positive findings (solid boxes)
        for f in positive_findings:
            x, y, w, h = f["bbox"]
            color = pred_colors[f["disease_idx"] % len(pred_colors)]
            ax_pred.add_patch(patches.Rectangle(
                (x, y), w, h, linewidth=2.5, edgecolor=color, facecolor=color, alpha=0.15))
            ax_pred.add_patch(patches.Rectangle(
                (x, y), w, h, linewidth=2.5, edgecolor=color, facecolor="none"))
            ax_pred.text(
                x, max(y - 4, 0),
                f"T{f['fdi_number']}: {f['disease']} ({f['confidence']:.0%})",
                fontsize=6.5, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.85),
                clip_on=True,
            )
            cx, cy = f["location_px"]
            ax_pred.plot(cx, cy, "w+", markersize=8, markeredgewidth=2)

        # Draw uncertain findings (dashed orange boxes with UNCERTAIN label)
        for f in uncertain_findings:
            x, y, w, h = f["bbox"]
            ax_pred.add_patch(patches.Rectangle(
                (x, y), w, h, linewidth=2.0, edgecolor=uncertain_color,
                facecolor=uncertain_color, alpha=0.08))
            ax_pred.add_patch(patches.Rectangle(
                (x, y), w, h, linewidth=2.0, edgecolor=uncertain_color,
                facecolor="none", linestyle="--"))
            ax_pred.text(
                x, max(y - 4, 0),
                f"UNCERTAIN: {f['disease']} ({f['confidence']:.0%})",
                fontsize=5.5, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=uncertain_color, alpha=0.75),
                clip_on=True,
            )
            cx, cy = f["location_px"]
            ax_pred.plot(cx, cy, "wx", markersize=6, markeredgewidth=1.5)

        ax_pred.axhline(y=img_size / 2, color="white", linestyle="--", alpha=0.35, linewidth=1)
        ax_pred.axvline(x=img_size / 2, color="white", linestyle="--", alpha=0.35, linewidth=1)
        for qx, qy, ql in [(img_size*.25, img_size*.06, "Q1"), (img_size*.75, img_size*.06, "Q2"),
                            (img_size*.75, img_size*.96, "Q3"), (img_size*.25, img_size*.96, "Q4")]:
            ax_pred.text(qx, qy, ql, fontsize=8, fontweight="bold", color="white", alpha=0.55, ha="center")
        n_pos = len(positive_findings)
        n_unc = len(uncertain_findings)
        label_parts = []
        if n_pos:
            label_parts.append(f"{n_pos} positive")
        if n_unc:
            label_parts.append(f"{n_unc} uncertain")
        status_str = ", ".join(label_parts) if label_parts else "No findings"
        diseases_str = ", ".join(sorted({f['disease'] for f in findings})) if findings else ""
        ax_pred.set_xlabel(
            f"{len(findings)} finding(s) ({status_str}): {diseases_str}"
            if findings else "No findings",
            fontsize=8, color="#CC2200",
        )
        ax_pred.axis("off")

        # Ground Truth panel 
        if gt_teeth is not None:
            ax_gt = axes[-1]
            ax_gt.imshow(display_img)
            ax_gt.set_title("Ground Truth (DENTEX)", fontsize=11, fontweight="bold", color="#007700")
            gt_disease_set = set()
            for gt in gt_teeth:
                x, y, w, h = gt["bbox"]
                x_d, y_d = x * scale_x, y * scale_y
                w_d, h_d = w * scale_x, h * scale_y
                ax_gt.add_patch(patches.Rectangle(
                    (x_d, y_d), w_d, h_d, linewidth=2.5, edgecolor="#00CC44",
                    facecolor="#00CC44", alpha=0.12, linestyle="--"))
                ax_gt.add_patch(patches.Rectangle(
                    (x_d, y_d), w_d, h_d, linewidth=2.5, edgecolor="#00CC44",
                    facecolor="none", linestyle="--"))
                gt_disease_set.add(gt["disease_name"])
                ax_gt.text(
                    x_d, max(y_d - 4, 0), gt["disease_name"],
                    fontsize=6.5, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#007700", alpha=0.85),
                    clip_on=True,
                )
            ax_gt.axhline(y=img_size / 2, color="white", linestyle="--", alpha=0.35, linewidth=1)
            ax_gt.axvline(x=img_size / 2, color="white", linestyle="--", alpha=0.35, linewidth=1)
            for qx, qy, ql in [(img_size*.25, img_size*.06, "Q1"), (img_size*.75, img_size*.06, "Q2"),
                                (img_size*.75, img_size*.96, "Q3"), (img_size*.25, img_size*.96, "Q4")]:
                ax_gt.text(qx, qy, ql, fontsize=8, fontweight="bold", color="white", alpha=0.55, ha="center")
            ax_gt.set_xlabel(
                f"{len(gt_teeth)} GT box(es): " + ", ".join(sorted(gt_disease_set))
                if gt_teeth else "No GT",
                fontsize=8, color="#007700",
            )
            ax_gt.axis("off")

        fig.suptitle(
            f"DentalScan AI - {os.path.basename(result['img_path'])}  "
            f"[GREEN=GT  COLOR=Predicted]",
            fontsize=12, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Visualization saved to: {save_path}")
        plt.show()
        return fig

    def cleanup(self):
        """Release GradCAM hooks."""
        self.gradcam.cleanup()


# ==================================================
# Helper: load GT tooth-level boxes from val manifest
# ==================================================
def load_gt_boxes_for_image(img_path, val_teeth_data, disease_names=None):
    """Return GT annotation boxes for a single image.

    Each item: {bbox: [x,y,w,h] (original pixels), disease_idx: int, disease_name: str}

    IMPORTANT: DENTEX 2023 has exactly 4 disease classes in its annotations.
    We always use the fixed DENTEX ground-truth label map.

    DENTEX GT classes (sorted alphabetically, matching original annotations):
      0: Caries
      1: Deep Caries
      2: Impacted
      3: Periapical Lesion

    Tries full path first, then basename fallback for path-format mismatches.
    """
    # Fixed DENTEX 4-class ground-truth map.
    DENTEX_GT = {0: "Caries", 1: "Deep Caries", 2: "Impacted", 3: "Periapical Lesion"}
    DENTEX_GT_NAME_TO_IDX = {v: k for k, v in DENTEX_GT.items()}

    gt = []
    if not val_teeth_data:
        return gt

    # 1) Exact path match
    entries = val_teeth_data.get(img_path, None)

    # 2) Basename fallback (handles /kaggle/input vs /kaggle/working mismatches)
    if entries is None:
        base = os.path.basename(img_path)
        for key, val in val_teeth_data.items():
            if os.path.basename(key) == base:
                entries = val
                break

    if not entries:
        return gt

    for tooth in entries:
        bbox = tooth.get("bbox")   # [x, y, w, h] in original pixels
        if bbox is None:
            continue

        # --- Resolve disease name (always use DENTEX string, NOT Phase-1 index) ---
        disease_name = tooth.get("disease_name", "").strip()

        # If stored as a recognised DENTEX class name, use directly
        if disease_name in DENTEX_GT_NAME_TO_IDX:
            disease_idx = DENTEX_GT_NAME_TO_IDX[disease_name]

        else:
            # Fall back to label vector / int
            raw_label = tooth.get("labels")
            if isinstance(raw_label, list):
                # Multi-hot vector in DENTEX 4-class ordering.
                dentex_labels = raw_label[:4]
                pos = [i for i, v in enumerate(dentex_labels) if v > 0]
                disease_idx = pos[0] if pos else None
            elif isinstance(raw_label, int) and raw_label < 4:
                disease_idx = raw_label
            else:
                # Can't determine — skip
                continue

        if disease_idx is None or disease_idx >= len(DENTEX_GT):
            continue

        # Use the canonical DENTEX disease name.
        disease_name = DENTEX_GT[disease_idx]

        gt.append({
            "bbox":         bbox,
            "disease_idx":  disease_idx,   # 0-3, DENTEX 4-class system
            "disease_name": disease_name,  # always one of the 4 DENTEX classes
        })
    return gt



# ==================================================
# 13. Initialize Pipeline
# ==================================================
pipeline = DentalVisionPipeline(
    model=model,
    config=CONFIG,
    gradcam_layer=CONFIG["cam_target_layer"],
)
print("\nDentalVisionPipeline initialized.")

# ==================================================
# 14. Demo: Process Val Split Images
# ==================================================
# Load val split manifest saved by Phase 1 (unseen images for honest evaluation)
val_manifest_path = None
for candidate in CONFIG.get("val_manifest_paths", []):
    if os.path.exists(candidate):
        val_manifest_path = candidate
        break

val_ground_truth = {}  # img_path -> image-level label vector
val_teeth_data = {}    # img_path -> list of {bbox, labels, disease_name}

if val_manifest_path:
    print(f"Loading val split manifest: {val_manifest_path}")
    with open(val_manifest_path, "r") as f:
        val_manifest = json.load(f)
    # Extract image paths and ground truth
    # Supports both formats:
    #   New: {"img_path": ..., "teeth": [{"bbox": ..., "labels": ...}, ...]}
    #   Legacy: {"img_path": ..., "labels": [...]}
    sample_images = []
    for entry in val_manifest:
        img_p = entry["img_path"]
        if not os.path.exists(img_p):
            continue
        sample_images.append(img_p)
        if "labels" in entry:
            val_ground_truth[img_p] = entry["labels"]
        elif "teeth" in entry:
            # Aggregate per-tooth class indices into an image-level multi-hot vector.
            img_label = np.zeros(CONFIG["num_classes"], dtype=np.float32)
            for tooth in entry["teeth"]:
                raw_label = tooth.get("labels")
                if isinstance(raw_label, int):
                    if 0 <= raw_label < CONFIG["num_classes"]:
                        img_label[raw_label] = 1.0
                elif isinstance(raw_label, list):
                    arr = np.array(raw_label, dtype=np.float32).reshape(-1)
                    if len(arr) == CONFIG["num_classes"]:
                        img_label = np.maximum(img_label, arr)
                    else:
                        pos = [i for i, v in enumerate(arr[:CONFIG["num_classes"]]) if v > 0]
                        for idx in pos:
                            img_label[idx] = 1.0
            val_ground_truth[img_p] = img_label.tolist()

        if "teeth" in entry:
            val_teeth_data[img_p] = entry["teeth"]  # Pass tooth bboxes to Phase 2
    print(f"  Found {len(sample_images)} val images (of {len(val_manifest)} in manifest)")
else:
    # Fallback: glob training folder if manifest not found
    print("WARNING: val_split_manifest.json not found. Searched:")
    for p in CONFIG.get("val_manifest_paths", []):
        print(f"  - {p}")
    print("Falling back to first 5 training images (NOT recommended).")
    img_dir = os.path.join(CONFIG["base_path"], "xrays")
    if os.path.exists(img_dir):
        sample_images = sorted(glob.glob(os.path.join(img_dir, "*.png")))[:5]
        if not sample_images:
            sample_images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:5]
    else:
        sample_images = []
        print(f"WARNING: Image directory not found: {img_dir}")

all_results = []
# Limit to 6 images, seeded for reproducibility
import random as _rng
DEMO_SEED = 99  # Change this to get different examples
_rng.seed(DEMO_SEED)
_rng.shuffle(sample_images)
sample_images = sample_images[:6]
print(f"Processing {len(sample_images)} images (seed={DEMO_SEED}, reproducible)")

for img_path in sample_images:
    print("\n" + "=" * 60)

    # Pass ground truth labels + tooth bboxes for two-stage inference
    gt_labels = val_ground_truth.get(img_path, None)
    gt_teeth  = val_teeth_data.get(img_path, None)
    if gt_teeth is None:
        # Basename fallback: handle path-format mismatches
        base = os.path.basename(img_path)
        for k in val_teeth_data:
            if os.path.basename(k) == base:
                gt_teeth = val_teeth_data[k]
                break
    print(f"  GT teeth entries for this image: {len(gt_teeth) if gt_teeth else 0}")
    result = pipeline.process(img_path, gt_labels=gt_labels, gt_teeth=gt_teeth)

    # Print ground truth in compact format
    if gt_labels is not None:
        gt_disease_names = [CONFIG["disease_names"][i] for i in range(CONFIG["num_classes"]) if i < len(gt_labels) and gt_labels[i] > 0.5]
        gt_display = ", ".join(gt_disease_names) if gt_disease_names else "None"
        # Only print if not already printed by two-stage mode
        if not val_teeth_data.get(img_path):
            print("\n--- Ground Truth ---")
            print(f"    Diseases: {gt_display}")
            print(f"    Raw labels: {gt_labels}")

    print("\n--- Generated Prompt ---")
    print(result["prompt"])
    print("--- End Prompt ---")

    # Save visualization
    save_name = f"phase2_analysis_{os.path.splitext(os.path.basename(img_path))[0]}.png"
    save_path = os.path.join(CONFIG["output_path"], save_name)
    pipeline.visualize(result, save_path=save_path)

    all_results.append(result)

# ==================================================
# 15. Summary Statistics
# ==================================================
if all_results:
    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY")
    print("=" * 60)

    total_findings = sum(len(r["findings"]) for r in all_results)
    images_with_findings = sum(1 for r in all_results if r["findings"])

    print(f"Images processed: {len(all_results)}")
    print(f"Images with findings: {images_with_findings}/{len(all_results)}")
    print(f"Total findings: {total_findings}")

    # Disease distribution across findings
    disease_counts = {}
    quadrant_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for r in all_results:
        for f in r["findings"]:
            disease_counts[f["disease"]] = disease_counts.get(f["disease"], 0) + 1
            quadrant_counts[f["quadrant"]] += 1

    print("\nDisease distribution in findings:")
    for disease, count in sorted(disease_counts.items()):
        print(f"  {disease}: {count}")

    print("\nQuadrant distribution:")
    quadrant_names = {1: "Q1 (upper right)", 2: "Q2 (upper left)",
                      3: "Q3 (lower left)", 4: "Q4 (lower right)"}
    for q in [1, 2, 3, 4]:
        print(f"  {quadrant_names[q]}: {quadrant_counts[q]}")

    # Save all prompts to a JSON file for Phase 3
    prompts_output = []
    for r in all_results:
        prompts_output.append({
            "img_path": r["img_path"],
            "prompt": r["prompt"],
            "findings": [
                {k: v for k, v in f.items() if k != "location_px" and k != "bbox"}
                for f in r["findings"]
            ],
            "probabilities": r["probabilities"].tolist(),
        })

    prompts_path = os.path.join(CONFIG["output_path"], "phase2_prompts.json")
    with open(prompts_path, "w") as f:
        json.dump(prompts_output, f, indent=2)
    print(f"\nPrompts saved to: {prompts_path}")

# ==================================================
# 16. Save Pipeline Config for Phase 3
# ==================================================
pipeline_meta = {
    "phase": 2,
    "description": "WBF Ensemble (DENTEX 2023 winner) + IoU Tooth Matching + GradCAM Visualization",
    "inference_mode": "wbf_ensemble" if CONFIG["use_wbf_ensemble"] else "two_stage_crop",
    "config": {k: v for k, v in CONFIG.items() if not isinstance(v, type)},
    "model_conf_threshold": CONFIG["model_conf_threshold"],
    "wbf_iou_threshold": CONFIG["wbf_iou_threshold"],
    "wbf_weights": CONFIG["wbf_weights"],
    "tooth_match_iou_threshold": CONFIG["tooth_match_iou_threshold"],
    "disease_names": CONFIG["disease_names"],
    "fdi_mapping_method": "iou_tooth_matching",
    "gradcam_role": "visualization_only",
    "gradcam_layer": CONFIG["cam_target_layer"],
}
meta_path = os.path.join(CONFIG["output_path"], "phase2_meta.json")
with open(meta_path, "w") as f:
    json.dump(pipeline_meta, f, indent=2)
print(f"Pipeline metadata saved to: {meta_path}")

print("\n" + "=" * 60)
print("Phase 2 complete.")
print(f"Inference mode: {'WBF Ensemble ' if CONFIG['use_wbf_ensemble'] else 'Two-stage crop classifier'}")
print("Outputs:")
print(f"  Visualizations: {CONFIG['output_path']}phase2_analysis_*.png")
print(f"  Prompts JSON:   {CONFIG['output_path']}phase2_prompts.json")
print(f"  Pipeline meta:  {CONFIG['output_path']}phase2_meta.json")

