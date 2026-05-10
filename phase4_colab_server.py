
import os
import io
import json
import base64
import random
import re
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from threading import Thread

os.environ["HF_HOME"] = "/content/drive/MyDrive/DentalScan/hf_cache"

# from google.colab import drive
# drive.mount('/content/drive')

# --------------------------------------------------
# Configuration
# --------------------------------------------------
CONFIG = {
    # Model paths -- Google Drive
    "vision_model_weights": "/content/drive/MyDrive/DentalScan/swin_s_dental_best_82.pth",
    "lora_adapter_path": "/content/drive/MyDrive/DentalScan/final_adapter",
    "llm_model_id": "NousResearch/Meta-Llama-3-8B-Instruct",

    # YOLO models -- Google Drive
    "yolo_tooth_model_path": "/content/drive/MyDrive/DentalScan/yolo_tooth_best.pt",
    "yolo_disease_model_path": "/content/drive/MyDrive/DentalScan/yolo_disease_best.pt",

    # Swin-S model settings 
    "num_classes": 4,
    "model_name": "swin_small_patch4_window7_224",
    "img_size": 336,
    "dropout_rate": 0.4,
    "disease_names": {0: "Caries", 1: "Deep Caries", 2: "Impacted", 3: "Periapical Lesion"},

    # Per-class decision thresholds 
    "class_thresholds": [0.60, 0.55, 0.50, 0.50],
    "class_threshold_adjustments": {
        "Caries": 0.0,
        "Deep Caries": 0.0,
        "Impacted": 0.0,
        "Periapical Lesion": -0.05,
    },
    "uncertain_margin": 0.1,
    "model_conf_threshold": 0.60,
    "cam_intensity_threshold": 0.05,
    "caries_suppression_threshold": 0.60,
    "max_findings_per_tooth": 2,
    "max_findings": 12,

    # WBF Ensemble
    "wbf_iou_threshold": 0.55,
    "wbf_skip_box_thr": 0.3,
    "wbf_weights": [1.0, 1.0],
    "wbf_conf_type": "avg",
    "wbf_final_min_conf_threshold": 0.60,
    "tooth_match_iou_threshold": 0.40,
    "tooth_crop_min_conf_threshold": 0.65,
    "yolo_conf_threshold": 0.45,
    "yolo_iou_threshold": 0.6,
    "yolo_disease_conf_threshold": 0.5,

    # TTA and temperature scaling
    "tta_enabled": True,
    "temperature": 1.5,

    # GradCAM 
    "cam_target_layer": "backbone.layers.3.blocks.1.norm2",

    # Inference mode: "llm" (full LLaMA 3) or "postprocess" (rule-based, instant)
    "inference_mode": "llm",

    # LLM inference settings (only used if inference_mode == "llm")
    "max_new_tokens": 512,
    "llm_temperature": 0.3,
    "top_p": 0.9,
    "repetition_penalty": 1.2,

    # Server
    "port": 8000,
    "ngrok_auth_token": "", 
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --------------------------------------------------
#  FDI Tooth Mapping & Disease Constants
# --------------------------------------------------
FDI_TEETH = {
    11: "Maxillary Right Central Incisor", 12: "Maxillary Right Lateral Incisor",
    13: "Maxillary Right Canine", 14: "Maxillary Right First Premolar",
    15: "Maxillary Right Second Premolar", 16: "Maxillary Right First Molar",
    17: "Maxillary Right Second Molar", 18: "Maxillary Right Third Molar",
    21: "Maxillary Left Central Incisor", 22: "Maxillary Left Lateral Incisor",
    23: "Maxillary Left Canine", 24: "Maxillary Left First Premolar",
    25: "Maxillary Left Second Premolar", 26: "Maxillary Left First Molar",
    27: "Maxillary Left Second Molar", 28: "Maxillary Left Third Molar",
    31: "Mandibular Left Central Incisor", 32: "Mandibular Left Lateral Incisor",
    33: "Mandibular Left Canine", 34: "Mandibular Left First Premolar",
    35: "Mandibular Left Second Premolar", 36: "Mandibular Left First Molar",
    37: "Mandibular Left Second Molar", 38: "Mandibular Left Third Molar",
    41: "Mandibular Right Central Incisor", 42: "Mandibular Right Lateral Incisor",
    43: "Mandibular Right Canine", 44: "Mandibular Right First Premolar",
    45: "Mandibular Right Second Premolar", 46: "Mandibular Right First Molar",
    47: "Mandibular Right Second Molar", 48: "Mandibular Right Third Molar",
}

SEVERITY_RANK = {"Periapical Lesion": 4, "Deep Caries": 3, "Impacted": 2, "Caries": 1}
ALLOWED_DISEASES = {"Caries", "Deep Caries", "Impacted", "Periapical Lesion"}

DISEASE_MERGE_MAP = {
    frozenset({"Deep Caries", "Periapical Lesion"}): "deep caries with associated periapical changes",
    frozenset({"Caries", "Periapical Lesion"}): "carious involvement with periapical pathology",
    frozenset({"Deep Caries", "Impacted"}): "deep caries on partially impacted tooth",
}

CERTAINTY_LANGUAGE = {
    "confirmed": ["indicative of", "consistent with", "strongly suggestive of"],
    "likely": ["consistent with", "suggestive of", "likely representing"],
    "suspected": ["suggestive of", "possibly representing", "raising concern for"],
}

REC_MERGE = {
    frozenset({"Vitality testing", "Endodontic treatment"}): "Vitality testing followed by possible endodontic treatment",
    frozenset({"Vitality testing", "Root canal therapy"}): "Vitality testing followed by possible endodontic treatment",
}

# --------------------------------------------------
#  Swin-S + CSRA Model Definition (synced from Phase 1/2)
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
    Must match the exact architecture used during Phase 1 training."""
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
        self.heads = nn.ModuleList([
            nn.Linear(backbone_dim, 1) for _ in range(num_classes)
        ])

    def forward(self, x):
        feature_map = self.backbone(x)  # Swin-S outputs (B, H, W, C) channels-last
        feature_map = feature_map.permute(0, 3, 1, 2)  # -> (B, C, H, W)
        gap_features = self.gap(feature_map).flatten(1)
        gap_features = self.dropout(gap_features)
        head_logits = torch.cat([h(gap_features) for h in self.heads], dim=1)
        csra_logits = self.csra(feature_map)
        return head_logits + csra_logits


# --------------------------------------------------
#  GradCAM Implementation (Swin-compatible, synced from Phase 2)
# --------------------------------------------------
class GradCAM:
    """GradCAM for Swin Transformer. Handles (B, N, C) token outputs
    from norm layers by reshaping to spatial (B, C, H, W) format."""

    def __init__(self, model, target_layer_name):
        self.model = model
        self.activations = None
        self.gradients = None
        self._hooks = []

        # Resolve target layer by name
        target_layer = None
        for name, module in model.named_modules():
            if name == target_layer_name:
                target_layer = module
                break
        if target_layer is None:
            raise ValueError(f"Target layer '{target_layer_name}' not found in model.")

        self._hooks.append(target_layer.register_forward_hook(self._forward_hook))
        self._hooks.append(target_layer.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        """Generate a GradCAM heatmap for a specific class."""
        self.model.zero_grad()
        logits = self.model(input_tensor)
        target_score = logits[0, class_idx]
        target_score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients.")

        activations = self.activations
        gradients = self.gradients

        # Swin norm layers output (B, N, C) tokens -> reshape to spatial
        if activations.dim() == 3:
            B, N, C = activations.shape
            h = w = int(N ** 0.5)
            activations = activations.permute(0, 2, 1).reshape(B, C, h, w)
            gradients = gradients.permute(0, 2, 1).reshape(B, C, h, w)

        weights = gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        cam_resized = cv2.resize(cam, (CONFIG["img_size"], CONFIG["img_size"]))
        return cam_resized

    def compute_cam_intensity(self, cam):
        """Compute meaningful GradCAM intensity using top-10% mean."""
        flat = cam.flatten()
        if flat.max() == 0:
            return 0.0
        top_k = max(1, int(len(flat) * 0.10))
        top_vals = np.sort(flat)[-top_k:]
        return float(np.mean(top_vals))

    def generate_all_classes(self, input_tensor, probabilities):
        """Generate heatmaps for all classes above model_conf_threshold."""
        heatmaps = {}
        for cls_idx in range(len(probabilities)):
            if probabilities[cls_idx] > CONFIG["model_conf_threshold"]:
                heatmap = self.generate(input_tensor, cls_idx)
                heatmaps[cls_idx] = heatmap
        return heatmaps

    def cleanup(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def create_heatmap_overlay(original_img, cam, alpha=0.5):
    """Overlay GradCAM heatmap on the original image."""
    img_array = np.array(original_img.resize((CONFIG["img_size"], CONFIG["img_size"])))
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(alpha * heatmap + (1 - alpha) * img_array)
    return Image.fromarray(overlay)


# --------------------------------------------------
#  Post-Processing (Rule-Based Report Generator)
# --------------------------------------------------
def build_clinical_impression(grouped_findings):
    """Generate specific clinical impressions based on disease patterns."""
    all_diseases = set()
    for tooth_findings in grouped_findings.values():
        for f in tooth_findings:
            all_diseases.add(f.get("disease", ""))

    if not grouped_findings:
        return "No radiographic evidence of pathology. Dentition within normal limits."

    impressions = []
    caries_count = sum(
        1 for fs in grouped_findings.values()
        for f in fs if f.get("disease") in ("Caries", "Deep Caries")
    )
    if caries_count >= 3:
        impressions.append("Generalized carious involvement across multiple teeth")
    elif caries_count > 0:
        impressions.append(f"Localized carious involvement ({caries_count} tooth/teeth affected)")
    if "Periapical Lesion" in all_diseases:
        impressions.append("Possible endodontic pathology requiring further evaluation")
    if "Impacted" in all_diseases:
        impressions.append("Eruption disturbance identified")
    if "Deep Caries" in all_diseases and "Periapical Lesion" in all_diseases:
        impressions.append("Advanced carious destruction with suspected pulpal or periapical sequelae")
    if not impressions:
        impressions.append("Findings noted; clinical correlation advised")

    return ". ".join(impressions) + "."


def postprocess_report(findings):
    """Generate a structured clinical report from findings using rule-based logic."""
    if not findings:
        return (
            "1. Summary\n"
            "No radiographic evidence of pathology. Dentition within normal limits.\n\n"
            "2. Clinical Impression\n"
            "The panoramic radiograph demonstrates no significant abnormalities. "
            "All visible teeth and supporting structures appear within normal radiographic limits.\n\n"
            "3. Recommendations\n"
            "- Routine follow-up and periodic radiographic monitoring advised.\n"
            "- Standard preventive care and oral hygiene maintenance recommended."
        )

    valid = [f for f in findings if f.get("disease") in ALLOWED_DISEASES]
    if not valid:
        return postprocess_report([])

    # Group by tooth
    tooth_groups = {}
    for f in valid:
        fdi = f.get("fdi_number", 0)
        if fdi not in tooth_groups:
            tooth_groups[fdi] = []
        tooth_groups[fdi].append(f)

    # Sort by severity
    sorted_teeth = sorted(
        tooth_groups.keys(),
        key=lambda fdi: max(SEVERITY_RANK.get(g["disease"], 0) for g in tooth_groups[fdi]),
        reverse=True,
    )

    summary_lines, detail_lines, rec_lines = [], [], []

    for fdi in sorted_teeth:
        group = tooth_groups[fdi]
        tooth_name = FDI_TEETH.get(fdi, f"Tooth {fdi}")
        diseases = [g["disease"] for g in group]
        disease_set = frozenset(diseases)

        best = max(group, key=lambda g: g.get("confidence", 0))
        certainty = best.get("certainty", "suspected")
        conf = best.get("confidence", 0.5)
        lang = random.choice(CERTAINTY_LANGUAGE.get(certainty, ["suggestive of"]))

        if disease_set in DISEASE_MERGE_MAP:
            merged_desc = DISEASE_MERGE_MAP[disease_set]
            summary_lines.append(f"{merged_desc.capitalize()} at tooth {fdi} ({tooth_name}).")
            detail_lines.append(
                f"Tooth {fdi} ({tooth_name}): Findings are {lang} {merged_desc}. "
                f"Confidence: {conf:.0%} ({certainty})."
            )
        elif len(diseases) > 1:
            combined = " with ".join(d.lower() for d in diseases)
            summary_lines.append(f"{combined.capitalize()} at tooth {fdi} ({tooth_name}).")
            detail_lines.append(
                f"Tooth {fdi} ({tooth_name}): Findings are {lang} {combined}. "
                f"Confidence: {conf:.0%} ({certainty})."
            )
        else:
            d = diseases[0]
            summary_lines.append(f"{d} ({certainty}) at tooth {fdi} ({tooth_name}).")
            detail_lines.append(
                f"Tooth {fdi} ({tooth_name}): Findings are {lang} {d.lower()}. "
                f"Confidence: {conf:.0%} ({certainty})."
            )

        # Merged recommendations
        tooth_recs = set()
        for g in group:
            d = g["disease"]
            if d == "Deep Caries":
                tooth_recs.update(["Vitality testing", "Endodontic treatment"])
            elif d == "Caries":
                tooth_recs.add("Restorative treatment")
            elif d == "Periapical Lesion":
                tooth_recs.update(["Endodontic treatment", "Vitality testing"])
            elif d == "Impacted":
                tooth_recs.add("Surgical evaluation")

        merged_rec = None
        for combo, merged in REC_MERGE.items():
            if combo.issubset(tooth_recs):
                remaining = tooth_recs - combo
                merged_rec = merged + ("; " + "; ".join(remaining) if remaining else "")
                break
        if not merged_rec:
            merged_rec = "; ".join(sorted(tooth_recs))
        rec_lines.append(f"Tooth {fdi}: {merged_rec}.")

    clinical_impression = build_clinical_impression(tooth_groups)

    report = "1. Summary\n"
    report += "Panoramic radiograph analysis reveals the following findings:\n"
    report += "\n".join(f"- {s}" for s in summary_lines)
    report += "\n\n2. Detailed Findings\n"
    report += "\n".join(f"- {d}" for d in detail_lines)
    report += f"\n\n3. Clinical Impression\n{clinical_impression}\n"
    report += "\n4. Recommendations\n"
    report += "\n".join(f"- {r}" for r in rec_lines)
    return report


# --------------------------------------------------
#  Image Preprocessing (albumentations, matches Phase 1/2 val_transform)
# --------------------------------------------------
val_transform = A.Compose([
    A.Resize(CONFIG["img_size"], CONFIG["img_size"]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# --------------------------------------------------
#  Load Models
# --------------------------------------------------
print("Loading Swin-S + CSRA model...")
vision_model = DentalCNN(
    model_name=CONFIG["model_name"],
    num_classes=CONFIG["num_classes"],
    pretrained=False,
    dropout_rate=CONFIG["dropout_rate"],
    img_size=CONFIG["img_size"],
)

if os.path.exists(CONFIG["vision_model_weights"]):
    checkpoint = torch.load(CONFIG["vision_model_weights"], map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        vision_model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Best mAP from training: {checkpoint.get('best_mAP', 'N/A')}")
    else:
        vision_model.load_state_dict(checkpoint)
    print("  Swin-S weights loaded successfully.")
else:
    print(f"  WARNING: Weights not found at {CONFIG['vision_model_weights']}")

vision_model = vision_model.to(DEVICE)
vision_model.eval()

# GradCAM target layer (Swin-S norm layer, resolved by name)
gradcam = GradCAM(vision_model, target_layer_name=CONFIG["cam_target_layer"])

# Load YOLO tooth detection model
yolo_tooth_model = None
if os.path.exists(CONFIG["yolo_tooth_model_path"]):
    from ultralytics import YOLO
    yolo_tooth_model = YOLO(CONFIG["yolo_tooth_model_path"])
    print(f"  YOLO tooth detector loaded: {CONFIG['yolo_tooth_model_path']}")
else:
    print(f"  WARNING: YOLO tooth model not found at {CONFIG['yolo_tooth_model_path']}")

# Load YOLO disease detection model
yolo_disease_model = None
if os.path.exists(CONFIG["yolo_disease_model_path"]):
    from ultralytics import YOLO as _YOLO
    yolo_disease_model = _YOLO(CONFIG["yolo_disease_model_path"])
    print(f"  YOLO disease detector loaded: {CONFIG['yolo_disease_model_path']}")
else:
    print(f"  WARNING: YOLO disease model not found at {CONFIG['yolo_disease_model_path']}")

# Optional: Load LLaMA 3 for full LLM inference
llm_model = None
llm_tokenizer = None

if CONFIG["inference_mode"] == "llm" and os.path.exists(CONFIG["lora_adapter_path"]):
    print("Loading LLaMA 3 with LoRA adapter...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    llm_tokenizer = AutoTokenizer.from_pretrained(CONFIG["lora_adapter_path"])
    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["llm_model_id"],
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )
    llm_model = PeftModel.from_pretrained(base_model, CONFIG["lora_adapter_path"])
    llm_model.eval()
    print("  LLaMA 3 loaded successfully.")
else:
    print(f"  Using '{CONFIG['inference_mode']}' mode for report generation.")


# --------------------------------------------------
#  WBF Ensemble Utilities (ported from Phase 2)
# --------------------------------------------------
def map_to_quadrant(x, y, w=336, h=336):
    """Map pixel coordinates to FDI dental quadrant (1-4)."""
    is_patient_right = x < w / 2
    is_upper = y < h / 2
    if is_patient_right and is_upper:
        return 1
    elif not is_patient_right and is_upper:
        return 2
    elif not is_patient_right and not is_upper:
        return 3
    return 4


def extract_fdi_tooth(x, quadrant, w=336):
    """Map x-coordinate + quadrant to a specific FDI tooth number (11-48)."""
    midline = w / 2
    if quadrant in [1, 4]:
        distance = (midline - x) / midline
    else:
        distance = (x - midline) / midline
    distance = max(0.0, min(1.0, distance))
    tooth_position = min(8, max(1, int(distance * 8) + 1))
    return quadrant * 10 + tooth_position


def get_fdi_tooth_name(fdi_number):
    """Return clinical name for an FDI tooth number."""
    return FDI_TEETH.get(fdi_number, f"Tooth {fdi_number}")


def interpret_confidence(conf):
    if conf > 0.85:
        return "confirmed"
    elif conf > 0.65:
        return "likely"
    return "suspected"


def map_severity(conf):
    if conf > 0.85:
        return "high"
    elif conf > 0.7:
        return "moderate"
    return "low"


def merge_detections_wbf(dino_boxes, dino_scores, dino_labels,
                          yolo_boxes, yolo_scores, yolo_labels,
                          img_w, img_h):
    """Merge disease detections from crop-classifier and YOLO using WBF."""
    from ensemble_boxes import weighted_boxes_fusion

    if not dino_boxes and not yolo_boxes:
        return [], [], []

    def xywh_to_norm_xyxy(boxes, w, h):
        result = []
        for x, y, bw, bh in boxes:
            result.append([max(0.0, x/w), max(0.0, y/h),
                           min(1.0, (x+bw)/w), min(1.0, (y+bh)/h)])
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
    merged_boxes_px = []
    for b in merged_boxes_norm:
        x1, y1, x2, y2 = b
        merged_boxes_px.append([x1*img_w, y1*img_h, (x2-x1)*img_w, (y2-y1)*img_h])
    return merged_boxes_px, merged_scores.tolist(), merged_labels.astype(int).tolist()


def compute_iou_xywh(box1, box2):
    """Compute IoU between two [x,y,w,h] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0]+box1[2], box2[0]+box2[2])
    y2 = min(box1[1]+box1[3], box2[1]+box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1, area2 = box1[2]*box1[3], box2[2]*box2[3]
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def get_tooth_boxes_with_fdi(img_path, img_w, img_h):
    """Run YOLO tooth detector and assign FDI numbers."""
    if yolo_tooth_model is None:
        return [], []
    results = yolo_tooth_model(
        img_path,
        conf=CONFIG.get("yolo_conf_threshold", 0.45),
        iou=CONFIG.get("yolo_iou_threshold", 0.6),
        verbose=False,
    )[0]
    tooth_boxes, tooth_fdi_ids = [], []
    disp = CONFIG["img_size"]
    if results.boxes is not None and len(results.boxes) > 0:
        sx, sy = disp / img_w, disp / img_h
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            dx1, dy1, dx2, dy2 = x1*sx, y1*sy, x2*sx, y2*sy
            tooth_boxes.append([dx1, dy1, dx2-dx1, dy2-dy1])
            cx, cy = (dx1+dx2)/2, (dy1+dy2)/2
            q = map_to_quadrant(cx, cy, w=disp, h=disp)
            tooth_fdi_ids.append(extract_fdi_tooth(cx, q, w=disp))
    return tooth_boxes, tooth_fdi_ids


def run_yolo_disease_detections(img_path, img_w, img_h):
    """Run YOLO disease detector, return display-scale xywh boxes."""
    if yolo_disease_model is None:
        return [], [], []
    results = yolo_disease_model(
        img_path, conf=CONFIG.get("yolo_disease_conf_threshold", 0.3), verbose=False
    )[0]
    boxes, scores, labels = [], [], []
    if results.boxes is not None and len(results.boxes) > 0:
        disp = CONFIG["img_size"]
        sx, sy = disp / img_w, disp / img_h
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes.append([float(x1*sx), float(y1*sy), float((x2-x1)*sx), float((y2-y1)*sy)])
            scores.append(float(box.conf[0].cpu()))
            labels.append(int(box.cls[0].cpu()))
    return boxes, scores, labels


def match_disease_to_tooth(disease_boxes, tooth_boxes, tooth_fdi_ids):
    """Match each disease bbox to a tooth bbox using IoU."""
    min_iou = CONFIG["tooth_match_iou_threshold"]
    matched = []
    for d_box in disease_boxes:
        best_iou, best_fdi, best_tbox = 0.0, None, None
        for t_box, fdi_id in zip(tooth_boxes, tooth_fdi_ids):
            iou = compute_iou_xywh(d_box, t_box)
            if iou > best_iou:
                best_iou, best_fdi, best_tbox = iou, fdi_id, t_box
        # Fallback to x-coordinate heuristic if no IoU match
        if best_fdi is None or best_iou < min_iou:
            cx = d_box[0] + d_box[2] / 2
            cy = d_box[1] + d_box[3] / 2
            q = map_to_quadrant(cx, cy, w=CONFIG["img_size"], h=CONFIG["img_size"])
            best_fdi = extract_fdi_tooth(cx, q, w=CONFIG["img_size"])
            best_iou = 0.0
        matched.append({"disease_box": d_box, "fdi_number": best_fdi,
                         "match_iou": best_iou, "tooth_box": best_tbox})
    return matched


def build_bridge_prompt(findings):
    """Build structured clinical prompt from findings (matches Phase 2/3 format)."""
    if not findings:
        return (
            "Patient Radiograph Findings:\n"
            "- No significant pathological findings detected.\n\n"
            "You are a dental radiology assistant.\n\n"
            "Generate a clinical report noting the absence of significant pathology.\n"
            "Format:\n1. Summary\n2. Clinical Impression\n3. Recommendations\n"
        )

    findings_sorted = sorted(findings, key=lambda f: f["fdi_number"])
    positive_findings = [f for f in findings_sorted if f.get("status") == "positive"]
    uncertain_findings = [f for f in findings_sorted if f.get("status") == "uncertain"]

    lines = []
    if positive_findings:
        lines.append("Confirmed Findings:")
        for f in positive_findings:
            lines.append(
                f"- {f['disease']} ({f['certainty']}, severity: {f['severity']}) "
                f"at tooth {f['fdi_number']} ({f['fdi_name']}) "
                f"with {f['confidence']:.0%} confidence"
            )
    if uncertain_findings:
        lines.append("\nUncertain Findings (require clinical verification):")
        for f in uncertain_findings:
            lines.append(
                f"- [UNCERTAIN] {f['disease']} at tooth {f['fdi_number']} "
                f"({f['fdi_name']}) with {f['confidence']:.0%} confidence "
                f"- borderline detection, recommend follow-up"
            )

    quadrant_names = {1: "upper right", 2: "upper left", 3: "lower left", 4: "lower right"}
    affected_quads = sorted(set(f.get("quadrant", 0) for f in findings))
    affected_text = ", ".join(quadrant_names.get(q, str(q)) for q in affected_quads)

    prompt = "Patient Radiograph Findings:\n"
    prompt += "\n".join(lines) + "\n\n"
    prompt += f"Affected regions: {affected_text}\n"
    prompt += f"Total findings: {len(findings)}\n\n"
    prompt += (
        "You are a dental radiology assistant.\n\n"
        "Generate a clinical report using ONLY the findings above.\n\n"
        "Rules:\n"
        "- Do NOT introduce new diseases or teeth not listed above.\n"
        "- Use cautious medical language: 'suggestive of' for suspected, "
        "'consistent with' for likely, 'indicative of' for confirmed.\n"
        "- Respect confidence levels: flag low-severity findings as requiring follow-up.\n\n"
        "Format:\n1. Summary\n2. Detailed Findings (tooth-wise)\n"
        "3. Clinical Impression\n4. Recommendations\n"
    )
    return prompt


# --------------------------------------------------
#  LLM Report Generation (used when inference_mode == "llm")
# --------------------------------------------------
SYSTEM_MESSAGE = (
    "You are a dental radiology assistant. You generate structured clinical "
    "reports from panoramic radiograph findings.\n\n"
    "STRICT RULES:\n"
    "1. You may ONLY report these four conditions: Caries, Deep Caries, Impacted, Periapical Lesion.\n"
    "2. You must NEVER introduce diseases or teeth not present in the input findings.\n"
    "3. If no findings are provided, state the dentition is within normal limits.\n"
    "4. Findings are categorized as 'Confirmed' (positive) or 'Uncertain' (borderline).\n"
    "   - Confirmed findings: report with full clinical detail.\n"
    "   - Uncertain findings: use cautious language and recommend clinical verification.\n"
    "5. Match your language to the confidence level:\n"
    "   - confirmed (>85%): use 'indicative of', 'consistent with'\n"
    "   - likely (65-85%): use 'suggestive of', 'likely representing'\n"
    "   - suspected (<65%): use 'possibly representing', 'raising concern for'\n"
    "6. Group multiple findings on the same tooth into a single entry.\n"
    "7. Order findings by clinical severity: Periapical Lesion > Deep Caries > Impacted > Caries.\n"
    "8. Merge related recommendations per tooth instead of listing them separately."
)


def format_llama3_chat(prompt, response=None):
    """Format a prompt into LLaMA 3 Instruct chat template."""
    text = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_MESSAGE}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    if response:
        text += f"{response}<|eot_id|>"
    return text


def generate_report_llm(prompt_text, findings):
    """Generate a clinical report using the fine-tuned LLaMA 3 model.

    Falls back to rule-based postprocess_report if the LLM is not loaded
    or if generation fails.
    """
    if llm_model is None or llm_tokenizer is None:
        print("  LLM not loaded, falling back to rule-based report.")
        return postprocess_report(findings)

    try:
        formatted = format_llama3_chat(prompt_text)
        inputs = llm_tokenizer(formatted, return_tensors="pt").to(llm_model.device)

        with torch.no_grad():
            output_ids = llm_model.generate(
                **inputs,
                max_new_tokens=CONFIG["max_new_tokens"],
                temperature=CONFIG["llm_temperature"],
                top_p=CONFIG["top_p"],
                repetition_penalty=CONFIG["repetition_penalty"],
                do_sample=True,
                pad_token_id=llm_tokenizer.eos_token_id,
            )

        # Decode only the new tokens (skip the input prompt)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_report = llm_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if not raw_report or len(raw_report) < 20:
            print("  LLM produced empty/short output, falling back to rule-based report.")
            return postprocess_report(findings)

        print(f"  LLM generated {len(raw_report)} chars of report text.")
        return raw_report

    except Exception as e:
        print(f"  LLM generation failed ({e}), falling back to rule-based report.")
        return postprocess_report(findings)



# --------------------------------------------------
#  Full Inference Pipeline
# --------------------------------------------------
def run_inference(image: Image.Image):
    """Full WBF ensemble pipeline (DENTEX 2023 winner approach).

    Pipeline flow:
      1. Save temp image for YOLO inference
      2. YOLO disease detector -> disease boxes (ensemble member 2)
      3. YOLO tooth detector -> tooth crops -> Swin-S classifier (member 1)
      4. WBF merges both detector outputs
      5. IoU-based tooth matching -> FDI numbers
      6. 3-tier threshold filtering (positive/uncertain/skip)
      7. GradCAM overlay (visualization only)
      8. Report generation (post-processor or LLM)

    Returns dict with: findings, bridge_prompt, generated_report, images, scores
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_np = np.array(image)
    img_h_orig, img_w_orig = img_np.shape[:2]
    disp = CONFIG["img_size"]

    # Save temp file for YOLO inference (YOLO needs a file path)
    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), "dental_input.jpg")
    image.save(tmp_path)

    # --- Member 2: YOLO disease detector ---
    yolo_boxes, yolo_scores, yolo_labels = run_yolo_disease_detections(
        tmp_path, img_w_orig, img_h_orig
    )
    print(f"  YOLO disease detector: {len(yolo_boxes)} detections")

    # --- Member 1: Swin-S crop classifier on YOLO tooth crops ---
    dino_boxes, dino_scores, dino_labels = [], [], []
    tooth_boxes_raw, _ = get_tooth_boxes_with_fdi(tmp_path, img_w_orig, img_h_orig)
    print(f"  YOLO tooth detector: {len(tooth_boxes_raw)} teeth found")

    class_thresholds = np.array(CONFIG["class_thresholds"])
    uncertain_margin = CONFIG["uncertain_margin"]
    sx, sy = img_w_orig / disp, img_h_orig / disp  # display -> original scale

    if tooth_boxes_raw:
        for t_box in tooth_boxes_raw:
            x, y, w, h = t_box
            # Extract crop from original image
            ox, oy, ow, oh = x * sx, y * sy, w * sx, h * sy
            pad = 0.15
            x1c = max(0, int(ox - ow * pad))
            y1c = max(0, int(oy - oh * pad))
            x2c = min(img_w_orig, int(ox + ow + ow * pad))
            y2c = min(img_h_orig, int(oy + oh + oh * pad))
            crop = img_np[y1c:y2c, x1c:x2c]
            if crop.shape[0] < 5 or crop.shape[1] < 5:
                continue
            aug = val_transform(image=crop)
            crop_tensor = aug["image"].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = vision_model(crop_tensor)
                if CONFIG.get("tta_enabled", True):
                    logits = (logits + vision_model(torch.flip(crop_tensor, dims=[3]))) / 2.0
                logits = logits / CONFIG.get("temperature", 1.5)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred_class = int(np.argmax(probs))
            pred_conf = float(probs[pred_class])

            # Per-class threshold check
            crop_threshold = float(class_thresholds[pred_class]) if pred_class < len(class_thresholds) else 0.5
            adj = CONFIG["class_threshold_adjustments"].get(
                CONFIG["disease_names"].get(pred_class, ""), 0.0)
            crop_threshold += adj

            if pred_conf >= crop_threshold - uncertain_margin:
                dino_boxes.append([x, y, w, h])
                dino_scores.append(pred_conf)
                dino_labels.append(pred_class)

    print(f"  Swin-S classifier (tooth crops): {len(dino_boxes)} disease detections")

    # --- WBF: merge both detector outputs ---
    merged_boxes, merged_scores, merged_labels = merge_detections_wbf(
        dino_boxes, dino_scores, dino_labels,
        yolo_boxes, yolo_scores, yolo_labels,
        img_w=disp, img_h=disp,
    )
    print(f"  WBF merged: {len(merged_boxes)} final disease detections")

    # --- IoU tooth matching ---
    tooth_boxes_disp, tooth_fdi_ids = get_tooth_boxes_with_fdi(
        tmp_path, img_w_orig, img_h_orig)
    matched = match_disease_to_tooth(merged_boxes, tooth_boxes_disp, tooth_fdi_ids)

    # --- Build findings with 3-tier thresholding ---
    findings = []
    threshold_adjustments = CONFIG.get("class_threshold_adjustments", {})
    caries_suppression = float(CONFIG.get("caries_suppression_threshold", 0.60))

    for m, score, label in zip(matched, merged_scores, merged_labels):
        label, score = int(label), float(score)
        disease_name = CONFIG["disease_names"].get(label, f"Class {label}")

        # Per-class threshold with adjustments
        if label < len(class_thresholds):
            report_threshold = float(class_thresholds[label])
        else:
            report_threshold = float(CONFIG["wbf_final_min_conf_threshold"])
        report_threshold += threshold_adjustments.get(disease_name, 0.0)
        if disease_name == "Caries":
            report_threshold = max(report_threshold, caries_suppression)

        # 3-tier decision
        if score >= report_threshold:
            status = "positive"
        elif score >= report_threshold - uncertain_margin:
            status = "uncertain"
        else:
            continue  # Skip

        d_box = m["disease_box"]
        fdi_number = m["fdi_number"]
        cx = d_box[0] + d_box[2] / 2
        cy = d_box[1] + d_box[3] / 2

        findings.append({
            "disease": disease_name,
            "disease_idx": label,
            "fdi_number": fdi_number,
            "fdi_name": get_fdi_tooth_name(fdi_number),
            "quadrant": map_to_quadrant(cx, cy, w=disp, h=disp),
            "confidence": score,
            "certainty": interpret_confidence(score),
            "severity": map_severity(score),
            "status": status,
            "cam_intensity": 0.0,
            "match_iou": m["match_iou"],
            "location_px": [int(cx), int(cy)],
            "bbox": [int(v) for v in d_box],
        })

    # Per-tooth limit
    max_per_tooth = int(CONFIG.get("max_findings_per_tooth", 2))
    tooth_groups = defaultdict(list)
    for f in findings:
        tooth_groups[f["fdi_number"]].append(f)
    findings = []
    for fdi, group in tooth_groups.items():
        group.sort(key=lambda f: f["confidence"], reverse=True)
        findings.extend(group[:max_per_tooth])

    # Cap at max_findings
    max_f = CONFIG.get("max_findings", 12)
    if len(findings) > max_f:
        findings = sorted(findings, key=lambda f: f["confidence"], reverse=True)[:max_f]

    findings.sort(key=lambda f: SEVERITY_RANK.get(f["disease"], 0), reverse=True)

    pos = sum(1 for f in findings if f["status"] == "positive")
    unc = sum(1 for f in findings if f["status"] == "uncertain")
    print(f"  Final findings: {len(findings)} ({pos} positive, {unc} uncertain)")

    # Build prompt and report
    bridge_prompt = build_bridge_prompt(findings)

    if CONFIG["inference_mode"] == "llm":
        report = generate_report_llm(bridge_prompt, findings)
    else:
        report = postprocess_report(findings)

    # --- Phase 2-style multi-panel visualization ---
    display_img = cv2.resize(img_np, (disp, disp))

    # Encode helper
    def img_to_base64(img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def np_to_base64(np_img):
        pil_img = Image.fromarray(np_img)
        return img_to_base64(pil_img)

    # Build list of visualization panels (Phase 2 style, no GT panel)
    viz_panels = []

    # Panel 1: Original X-ray
    viz_panels.append({
        "title": "Original X-ray",
        "image": np_to_base64(display_img),
        "badge": None,
    })

    composite_heatmaps = {}
    if findings:
        for f in findings:
            cls_idx = f["disease_idx"]
            x, y, w, h = f["bbox"]
            try:
                # Extract crop from display image at disease bbox location
                x1c = max(0, int(x))
                y1c = max(0, int(y))
                x2c = min(disp, int(x + w))
                y2c = min(disp, int(y + h))
                crop_disp = display_img[y1c:y2c, x1c:x2c]
                if crop_disp.shape[0] < 5 or crop_disp.shape[1] < 5:
                    continue
                # Preprocess crop for model (same as Phase 2)
                crop_tensor = val_transform(image=crop_disp)["image"].unsqueeze(0).to(DEVICE)
                # Run GradCAM on crop for this disease class
                cam = gradcam.generate(crop_tensor, cls_idx)
                # Resize cam back to crop dimensions and composite onto full canvas
                cw = max(1, x2c - x1c)
                ch = max(1, y2c - y1c)
                cam_resized = cv2.resize(cam, (cw, ch))
                if cls_idx not in composite_heatmaps:
                    composite_heatmaps[cls_idx] = np.zeros((disp, disp), dtype=np.float32)
                region = composite_heatmaps[cls_idx][y1c:y1c+ch, x1c:x1c+cw]
                if region.shape == cam_resized.shape:
                    composite_heatmaps[cls_idx][y1c:y1c+ch, x1c:x1c+cw] = np.maximum(region, cam_resized)
            except Exception:
                pass  # GradCAM is optional, never blocks findings

    #  only heatmaps with actual activations
    composite_heatmaps = {k: v for k, v in composite_heatmaps.items() if v.max() > 0.01}

    # Build GradCAM overlay panels from composite heatmaps
    detected_classes = sorted(
        composite_heatmaps.keys(),
        key=lambda c: max(
            (f["confidence"] for f in findings if f["disease_idx"] == c), default=0
        ),
        reverse=True,
    )
    for cls_idx in detected_classes[:3]:  # Cap at 3 heatmap panels
        cam = composite_heatmaps[cls_idx]
        heatmap_colored = cv2.applyColorMap(
            (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(display_img, 0.5, heatmap_colored, 0.5, 0)
        disease_name = CONFIG["disease_names"].get(cls_idx, f"Class {cls_idx}")
        cls_conf = max(
            (f["confidence"] for f in findings if f["disease_idx"] == cls_idx),
            default=0
        )
        viz_panels.append({
            "title": f"GradCAM: {disease_name}",
            "image": np_to_base64(overlay),
            "badge": f"conf={cls_conf:.0%}",
        })

    # Final panel: Predicted Findings (annotated bounding boxes)
    pred_img = display_img.copy()
    pred_colors = {
        0: (255, 107, 107),  # Caries - red
        1: (78, 205, 196),   # Deep Caries - teal
        2: (255, 230, 109),  # Impacted - yellow
        3: (168, 230, 207),  # Periapical - green
    }
    uncertain_color = (255, 165, 0)  # Orange

    positive_findings = [f for f in findings if f.get("status") == "positive"]
    uncertain_findings = [f for f in findings if f.get("status") == "uncertain"]

    for f in positive_findings:
        x, y, w, h = f["bbox"]
        color = pred_colors.get(f["disease_idx"], (200, 200, 200))
        # Draw filled semi-transparent rectangle
        overlay_rect = pred_img.copy()
        cv2.rectangle(overlay_rect, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay_rect, 0.15, pred_img, 0.85, 0, pred_img)
        # Draw border
        cv2.rectangle(pred_img, (x, y), (x + w, y + h), color, 2)
        # Label
        label_text = f"T{f['fdi_number']}: {f['disease']} ({f['confidence']:.0%})"
        font_scale, thickness = 0.35, 1
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_y = max(y - 4, th + 4)
        cv2.rectangle(pred_img, (x, label_y - th - 4), (x + tw + 6, label_y + 2), color, -1)
        cv2.putText(pred_img, label_text, (x + 3, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        # Center cross
        cx, cy = f["location_px"]
        cv2.drawMarker(pred_img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 10, 2)

    for f in uncertain_findings:
        x, y, w, h = f["bbox"]
        # Dashed border (simulate with dotted lines)
        for i in range(0, w, 8):
            cv2.line(pred_img, (x + i, y), (x + min(i + 4, w), y), uncertain_color, 2)
            cv2.line(pred_img, (x + i, y + h), (x + min(i + 4, w), y + h), uncertain_color, 2)
        for i in range(0, h, 8):
            cv2.line(pred_img, (x, y + i), (x, y + min(i + 4, h)), uncertain_color, 2)
            cv2.line(pred_img, (x + w, y + i), (x + w, y + min(i + 4, h)), uncertain_color, 2)
        label_text = f"UNCERTAIN: {f['disease']} ({f['confidence']:.0%})"
        font_scale, thickness = 0.3, 1
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_y = max(y - 4, th + 4)
        cv2.rectangle(pred_img, (x, label_y - th - 4), (x + tw + 6, label_y + 2), uncertain_color, -1)
        cv2.putText(pred_img, label_text, (x + 3, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cx, cy = f["location_px"]
        cv2.drawMarker(pred_img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 8, 1)

    # Draw quadrant lines
    cv2.line(pred_img, (disp // 2, 0), (disp // 2, disp), (255, 255, 255), 1)
    cv2.line(pred_img, (0, disp // 2), (disp, disp // 2), (255, 255, 255), 1)
    for qx, qy, ql in [(disp // 4, 18, "Q1"), (3 * disp // 4, 18, "Q2"),
                        (3 * disp // 4, disp - 8, "Q3"), (disp // 4, disp - 8, "Q4")]:
        cv2.putText(pred_img, ql, (qx - 8, qy), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

    n_pos = len(positive_findings)
    n_unc = len(uncertain_findings)
    pred_badge = f"{len(findings)} finding(s)"
    if n_pos:
        pred_badge += f" ({n_pos} positive"
        if n_unc:
            pred_badge += f", {n_unc} uncertain"
        pred_badge += ")"
    elif n_unc:
        pred_badge += f" ({n_unc} uncertain)"

    viz_panels.append({
        "title": "Predicted Findings",
        "image": np_to_base64(pred_img),
        "badge": pred_badge if findings else "No findings",
    })

    # Legacy compatibility: keep original_image and heatmap_image fields
    original_b64 = np_to_base64(display_img)
    heatmap_b64 = viz_panels[1]["image"] if len(viz_panels) > 1 else original_b64

    # Cleanup temp file
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    return {
        "findings": findings,
        "bridge_prompt": bridge_prompt,
        "generated_report": report,
        "original_image": original_b64,
        "heatmap_image": heatmap_b64,
        "viz_panels": viz_panels,
        "top_prediction": findings[0]["disease"] if findings else "No pathology detected",
        "confidence_scores": {f["disease"]: f["confidence"] for f in findings},
    }


# --------------------------------------------------
#  FastAPI Application
# --------------------------------------------------
app = FastAPI(title="DentalScan AI - Inference Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok", "device": str(DEVICE), "mode": CONFIG["inference_mode"]}


@app.post("/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    """Analyze a dental X-ray image and return findings + report."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = run_inference(image)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# --------------------------------------------------
#  Start Server with ngrok
# --------------------------------------------------
def start_server():
    """Start FastAPI server and expose via ngrok."""
    from pyngrok import ngrok

    # Get auth token: Colab Secrets > CONFIG > manual input
    auth_token = CONFIG["ngrok_auth_token"]
    if not auth_token:
        try:
            from google.colab import userdata
            auth_token = userdata.get('ngrok')
            print("  ngrok token loaded from Colab Secrets.")
        except Exception:
            auth_token = input("Enter your ngrok auth token: ").strip()
    if auth_token:
        ngrok.set_auth_token(auth_token)

    # Open ngrok tunnel
    public_url = ngrok.connect(CONFIG["port"])
    print("=" * 60)
    print("DentalScan AI Server is LIVE")
    print("=" * 60)
    print(f"  Local URL:  http://localhost:{CONFIG['port']}")
    print(f"  Public URL: {public_url}")
    print(f"  Mode:       {CONFIG['inference_mode']}")
    print("=" * 60)
    print(f"\nCopy this URL into your web app: {public_url}")
    print("Press Ctrl+C to stop.\n")

    # Run uvicorn in a thread so Colab doesn't block
    thread = Thread(target=uvicorn.run, args=(app,), kwargs={"host": "0.0.0.0", "port": CONFIG["port"]})
    thread.daemon = True
    thread.start()

    return public_url


public_url = start_server()
