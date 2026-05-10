
import warnings
warnings.filterwarnings("ignore")

import json
import os
import shutil
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# --------------------------------------------------
# 1. CONFIG
# --------------------------------------------------
CONFIG = {
    "seed": 42,
    "img_size": 672,           
    "batch_size": 16,           # Paper batch size
    "epochs": 100,              # Paper: 100 epochs
    "patience": 20,            
    "model_variant": "yolov8x.pt",  
    # Single class: "tooth" 
    "num_classes": 1,
    "class_names": ["tooth"],
    # Preprocessing 
    "remove_borders": True,     # Remove black scanner borders
    "apply_clahe": True,        # CLAHE histogram equalization
    "clahe_clip_limit": 2.0,    # CLAHE clip limit
    "clahe_grid_size": (8, 8),  # CLAHE tile grid size
   
    "data_sources": [
        {
            "name": "quadrant_enumeration",
            "base_path": "/kaggle/input/datasets/truthisneverlinear/dentex-challenge-2023/training_data/training_data/quadrant_enumeration",
            "json_file": "train_quadrant_enumeration.json",
            "img_subdir": "xrays",
        },
        {
            "name": "quadrant_enumeration_disease",
            "base_path": "/kaggle/input/datasets/truthisneverlinear/dentex-challenge-2023/training_data/training_data/quadrant-enumeration-disease",
            "json_file": "train_quadrant_enumeration_disease.json",
            "img_subdir": "xrays",
        },
    ],
    "output_path": "/kaggle/working/yolo_tooth_detection",
   
    "val_split": 0.2,
}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

seed_everything(CONFIG["seed"])

print("=" * 60)
print("Phase 0.5: YOLO Tooth Detection Training (Paper-Aligned)")
print("=" * 60)
print(f"Model: {CONFIG['model_variant']}")
print(f"Image size: {CONFIG['img_size']}px")
print(f"Preprocessing: border_removal={CONFIG['remove_borders']}, CLAHE={CONFIG['apply_clahe']}")

# --------------------------------------------------
# 2. Preprocessing Functions
# --------------------------------------------------
def remove_black_borders(img):
    """Remove black borders from panoramic X-ray (scanner artifact).

    Paper: "These borders were removed using an algorithm that scanned the
    images in multiple directions to detect where the borders ended."

    Returns:
        cropped_img: Image with borders removed
        offset: (x_offset, y_offset) for bbox adjustment
        scale: (x_scale, y_scale) - always (1,1) since no resize here
    """
    if img is None:
        return img, (0, 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    # Threshold to find non-black regions
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find bounding rect of all non-black pixels
    coords = cv2.findNonZero(thresh)
    if coords is not None and len(coords) > 0:
        x, y, w, h = cv2.boundingRect(coords)
        # Only crop if border is significant (>2% of image dimension)
        img_h, img_w = img.shape[:2]
        if x > img_w * 0.02 or y > img_h * 0.02 or \
           (img_w - (x + w)) > img_w * 0.02 or (img_h - (y + h)) > img_h * 0.02:
            return img[y:y+h, x:x+w], (x, y)
    return img, (0, 0)


def apply_clahe_enhancement(img, clip_limit=2.0, grid_size=(8, 8)):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Paper: "histogram equalization was applied to improve the contrast
    and intensity distribution of the images."

    Using CLAHE instead of global HE for better local contrast.
    """
    if img is None:
        return img

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced = clahe.apply(gray)

    # Convert back to BGR (YOLO expects 3-channel)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def preprocess_image(img_path, remove_borders=True, use_clahe=True,
                     clip_limit=2.0, grid_size=(8, 8)):
    """Full preprocessing pipeline matching the paper.

    Returns:
        processed_img: Preprocessed image
        offset: (x, y) offset from border removal (for bbox adjustment)
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, (0, 0)

    offset = (0, 0)

    # Step 1: Remove black borders
    if remove_borders:
        img, offset = remove_black_borders(img)

    # Step 2: CLAHE histogram equalization
    if use_clahe:
        img = apply_clahe_enhancement(img, clip_limit, grid_size)

    return img, offset


# --------------------------------------------------
# 3. Load DENTEX Annotations (Multiple Sources)
# --------------------------------------------------
print("\n--- Loading Annotations ---")

# Collect all images and annotations across data sources
all_images = {}     # filename -> {img_info, annotations, source_path}
all_annotations = {}  # filename -> [annotations]

for source in CONFIG["data_sources"]:
    json_path = os.path.join(source["base_path"], source["json_file"])

    if not os.path.exists(json_path):
        print(f"  SKIP: {source['name']} - JSON not found: {json_path}")
        continue

    with open(json_path, "r") as f:
        data = json.load(f)

    img_dir = os.path.join(source["base_path"], source["img_subdir"])
    id_to_info = {img["id"]: img for img in data["images"]}
    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    # Group annotations by image
    source_anns = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in source_anns:
            source_anns[img_id] = []
        source_anns[img_id].append(ann)

    # Merge into global dict (dedup by filename)
    added = 0
    for img_id, img_info in id_to_info.items():
        filename = img_info["file_name"]
        img_path = os.path.join(img_dir, filename)

        if not os.path.exists(img_path):
            continue
        if img_id not in source_anns:
            continue

        if filename not in all_images:
            all_images[filename] = {
                "info": img_info,
                "img_dir": img_dir,
                "annotations": source_anns[img_id],
            }
            added += 1
        else:
            # Merge annotations (take the set with more annotations)
            existing_count = len(all_images[filename]["annotations"])
            new_count = len(source_anns[img_id])
            if new_count > existing_count:
                all_images[filename]["annotations"] = source_anns[img_id]
                all_images[filename]["img_dir"] = img_dir

    print(f"  {source['name']}: loaded {added} new images from {json_path}")

filenames = list(all_images.keys())
print(f"\nTotal unique images: {len(filenames)}")

# Stats
teeth_counts = [len(all_images[fn]["annotations"]) for fn in filenames]
print(f"Teeth per image: min={min(teeth_counts)}, max={max(teeth_counts)}, "
      f"mean={np.mean(teeth_counts):.1f}, total={sum(teeth_counts)}")

# --------------------------------------------------
# 4. Convert COCO Annotations to YOLO Format
# --------------------------------------------------
def coco_to_yolo(bbox, img_width, img_height, offset=(0, 0)):
    """Convert COCO bbox [x, y, w, h] to YOLO [x_center, y_center, w, h] normalized.

    Args:
        bbox: [x_topleft, y_topleft, width, height] in pixels
        img_width, img_height: dimensions of the (possibly cropped) image
        offset: (x_offset, y_offset) from border removal
    """
    x, y, w, h = bbox
    # Adjust for border removal offset
    x -= offset[0]
    y -= offset[1]

    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.001, min(1.0, w_norm))
    h_norm = max(0.001, min(1.0, h_norm))

    # Validate: bbox must be within image after offset adjustment
    if x_center - w_norm / 2 < -0.1 or y_center - h_norm / 2 < -0.1:
        return None 
    if x_center + w_norm / 2 > 1.1 or y_center + h_norm / 2 > 1.1:
        return None

    return x_center, y_center, w_norm, h_norm

# --------------------------------------------------
# 5. Train/Val Split (Image-Level)
# --------------------------------------------------
from sklearn.model_selection import train_test_split

train_files, val_files = train_test_split(
    filenames,
    test_size=CONFIG["val_split"],
    random_state=CONFIG["seed"],
)

print(f"\nSplit: {len(train_files)} train | {len(val_files)} val")

# --------------------------------------------------
# 6. Create YOLO Dataset with Preprocessing
# --------------------------------------------------
dataset_root = CONFIG["output_path"]
for split in ["train", "val"]:
    os.makedirs(os.path.join(dataset_root, "images", split), exist_ok=True)
    os.makedirs(os.path.join(dataset_root, "labels", split), exist_ok=True)

print(f"\nYOLO dataset directory: {dataset_root}")
print(f"Preprocessing: border_removal={CONFIG['remove_borders']}, CLAHE={CONFIG['apply_clahe']}")

skipped = 0
total_teeth = 0
border_removed_count = 0

for split, file_list in [("train", train_files), ("val", val_files)]:
    split_teeth = 0
    for filename in file_list:
        entry = all_images[filename]
        img_info = entry["info"]
        img_dir = entry["img_dir"]
        annotations = entry["annotations"]
        img_path = os.path.join(img_dir, filename)

        # --- Preprocess image ---
        processed_img, offset = preprocess_image(
            img_path,
            remove_borders=CONFIG["remove_borders"],
            use_clahe=CONFIG["apply_clahe"],
            clip_limit=CONFIG["clahe_clip_limit"],
            grid_size=CONFIG["clahe_grid_size"],
        )

        if processed_img is None:
            skipped += 1
            continue

        if offset != (0, 0):
            border_removed_count += 1

        # Get processed image dimensions
        proc_h, proc_w = processed_img.shape[:2]

        # Save preprocessed image
        dst_img = os.path.join(dataset_root, "images", split, filename)
        cv2.imwrite(dst_img, processed_img)

        # Create YOLO label file with adjusted bboxes
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(dataset_root, "labels", split, label_filename)

        with open(label_path, "w") as f:
            for ann in annotations:
                bbox = ann.get("bbox")
                if bbox is None:
                    continue
                x, y, w, h = bbox
                if w < 5 or h < 5:  # Skip tiny annotations
                    continue

                result = coco_to_yolo(bbox, proc_w, proc_h, offset)
                if result is None:
                    continue  # bbox outside cropped region

                x_c, y_c, w_n, h_n = result
                # Class 0 = tooth (single class detection)
                f.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
                split_teeth += 1

    total_teeth += split_teeth
    print(f"  {split}: {len(file_list)} images, {split_teeth} tooth annotations")

print(f"  Total teeth: {total_teeth} | Skipped: {skipped} | Borders removed: {border_removed_count}")

# --------------------------------------------------
# 7. Create data.yaml for YOLO
# --------------------------------------------------
data_yaml_content = f"""# DentalScan YOLO Tooth Detection Dataset
# Auto-generated by phase0_5_yolo_tooth_detection.py
# Methodology: Mendes et al. (2025) - YOLO for panoramic radiographs

path: {os.path.abspath(dataset_root)}
train: images/train
val: images/val

nc: {CONFIG['num_classes']}
names: {CONFIG['class_names']}
"""

data_yaml_path = os.path.join(dataset_root, "data.yaml")
with open(data_yaml_path, "w") as f:
    f.write(data_yaml_content)

print(f"\ndata.yaml saved to: {data_yaml_path}")

# --------------------------------------------------
# 8. Visualize Preprocessed Samples 
# --------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

sample_files = random.sample(train_files, min(8, len(train_files)))

for ax_idx, filename in enumerate(sample_files):
    # Load preprocessed image from YOLO dataset dir
    img_path = os.path.join(dataset_root, "images", "train", filename)
    img = cv2.imread(img_path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load YOLO labels
    label_path = os.path.join(dataset_root, "labels", "train",
                              os.path.splitext(filename)[0] + ".txt")
    img_h, img_w = img.shape[:2]

    ax = axes[ax_idx]
    ax.imshow(img_rgb)

    n_teeth = 0
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, xc, yc, wn, hn = [float(p) for p in parts]
                # Convert back to pixel coords for visualization
                x1 = (xc - wn/2) * img_w
                y1 = (yc - hn/2) * img_h
                w = wn * img_w
                h = hn * img_h
                rect = plt.Rectangle((x1, y1), w, h, linewidth=1.5,
                                      edgecolor="cyan", facecolor="none")
                ax.add_patch(rect)
                n_teeth += 1

    ax.set_title(f"{filename[:20]}...\n{n_teeth} teeth | {img_w}x{img_h}", fontsize=8)
    ax.axis("off")

plt.suptitle("Preprocessed YOLO Training Data (Border Removed + CLAHE)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_path"], "preprocessed_samples.png"), dpi=150)
plt.show()

# --------------------------------------------------
# 9. Train YOLOv8x 
# --------------------------------------------------
print("\n" + "=" * 60)
print("Starting YOLOv8x Training (Paper-Aligned)")
print("=" * 60)

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    subprocess.check_call(["pip", "install", "ultralytics", "-q"])
    from ultralytics import YOLO


model = YOLO(CONFIG["model_variant"])
print(f"Loaded base model: {CONFIG['model_variant']}")


results = model.train(
    data=data_yaml_path,
    epochs=CONFIG["epochs"],
    imgsz=CONFIG["img_size"],
    batch=CONFIG["batch_size"],
    patience=CONFIG["patience"],
    save=True,
    save_period=-1,  
    project=CONFIG["output_path"],
    name="train",
    exist_ok=True,
   
    # We keep only light,medically-appropriate augmentations
    hsv_h=0.0,             
    hsv_s=0.0,             
    hsv_v=0.15,             
    degrees=3.0,            
    translate=0.1,          
    scale=0.2,              
    fliplr=0.5,             
    flipud=0.0,             
    mosaic=0.0,             
    mixup=0.0,              
    copy_paste=0.0,       
    erasing=0.0,            
    # ---- Dense tooth detection tuning ----
    box=7.5,                # Bbox regression loss weight
    cls=0.5,                # Classification loss weight
    iou=0.6,                # NMS IoU threshold
    max_det=50,             # Max detections per image (teeth + margin)
    # ---- Optimizer ----
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
    # ---- Device ----
    device=0 if os.path.exists("/dev/nvidia0") or os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    verbose=True,
)


last_pt = os.path.join(CONFIG["output_path"], "train", "weights", "last.pt")
if os.path.exists(last_pt):
    os.remove(last_pt)
    print("Deleted last.pt (keeping best.pt only)")

# --------------------------------------------------
# 10. Evaluate Best Model
# --------------------------------------------------
print("\n" + "=" * 60)
print("Evaluating Best Model")
print("=" * 60)

# Find best model
best_model_path = os.path.join(CONFIG["output_path"], "train", "weights", "best.pt")
if not os.path.exists(best_model_path):
    for root, dirs, files in os.walk(CONFIG["output_path"]):
        for f in files:
            if f == "best.pt":
                best_model_path = os.path.join(root, f)
                break

print(f"Best model: {best_model_path}")

best_model = YOLO(best_model_path)
val_results = best_model.val(
    data=data_yaml_path,
    imgsz=CONFIG["img_size"],
    batch=CONFIG["batch_size"],
    verbose=True,
)

# Print key metrics
print("\n" + "=" * 40)
print("VALIDATION METRICS")
print("=" * 40)
print(f"  Precision:  {val_results.box.mp:.4f}")
print(f"  Recall:     {val_results.box.mr:.4f}")
print(f"  mAP@50:     {val_results.box.map50:.4f}")
print(f"  mAP@50-95:  {val_results.box.map:.4f}")
print(f"\n  Paper target (YOLOv8x):")
print(f"  Precision:  0.9283")
print(f"  Recall:     0.9327")
print(f"  mAP@50:     0.9450")
print(f"  mAP@50-95:  0.5781")

# Gap analysis
gap_p = val_results.box.mp - 0.9283
gap_r = val_results.box.mr - 0.9327
gap_map50 = val_results.box.map50 - 0.9450
gap_map = val_results.box.map - 0.5781
print(f"\n  Delta vs paper:")
print(f"  Precision:  {gap_p:+.4f}")
print(f"  Recall:     {gap_r:+.4f}")
print(f"  mAP@50:     {gap_map50:+.4f}")
print(f"  mAP@50-95:  {gap_map:+.4f}")

# --------------------------------------------------
# 11. Inference Demo on Val Images
# --------------------------------------------------
print("\n" + "=" * 60)
print("Inference Demo")
print("=" * 60)

fig, axes = plt.subplots(2, 4, figsize=(22, 11))
axes = axes.flatten()

demo_files = random.sample(val_files, min(8, len(val_files)))

for ax_idx, filename in enumerate(demo_files):
    # Use preprocessed image
    img_path = os.path.join(dataset_root, "images", "val", filename)
    img = cv2.imread(img_path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]

    # Run inference
    preds = best_model(img_path, imgsz=CONFIG["img_size"], verbose=False)[0]
    boxes = preds.boxes

    ax = axes[ax_idx]
    ax.imshow(img_rgb)

    # Draw GT boxes (green dashed)
    label_path = os.path.join(dataset_root, "labels", "val",
                              os.path.splitext(filename)[0] + ".txt")
    n_gt = 0
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, xc, yc, wn, hn = [float(p) for p in parts]
                x1 = (xc - wn/2) * img_w
                y1 = (yc - hn/2) * img_h
                w = wn * img_w
                h = hn * img_h
                rect = plt.Rectangle((x1, y1), w, h, linewidth=1.5,
                                      edgecolor="lime", facecolor="none", linestyle="--")
                ax.add_patch(rect)
                n_gt += 1

    # Draw predicted boxes (cyan solid)
    n_pred = 0
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            if conf < 0.25:
                continue
            w = x2 - x1
            h = y2 - y1
            rect = plt.Rectangle((x1, y1), w, h, linewidth=2,
                                  edgecolor="cyan", facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1 - 4, f"{conf:.2f}", fontsize=6, color="cyan",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.7))
            n_pred += 1

    ax.set_title(f"{filename[:18]}...\nGT: {n_gt} | Pred: {n_pred}", fontsize=8)
    ax.axis("off")

plt.suptitle("YOLOv8x Tooth Detection - GT (green) vs Predictions (cyan)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_path"], "detection_demo.png"), dpi=150)
plt.show()

# --------------------------------------------------
# 12. Copy Best Model to Working Directory
# --------------------------------------------------
final_model_path = "/kaggle/working/yolo_tooth_best.pt"
shutil.copy2(best_model_path, final_model_path)
print(f"\nBest model copied to: {final_model_path}")

# Summary
print("\n" + "=" * 60)
print("Phase 0.5 Complete (Paper-Aligned)")
print("=" * 60)
print(f"Model:       {CONFIG['model_variant']} (YOLOv8 extra-large)")
print(f"Image size:  {CONFIG['img_size']}px")
print(f"Preprocess:  border_removal + CLAHE")
print(f"Augments:    mosaic=OFF, mixup=OFF (paper methodology)")
print(f"Images:      {len(train_files)} train + {len(val_files)} val")
print(f"Teeth:       {total_teeth} annotations")
print(f"Best model:  {final_model_path}")



# ==============================================================
# PART 2: YOLO Disease Detection 
# ==============================================================


print("\n" + "=" * 60)
print("PART 2: YOLOv8 Disease Detection Training (2nd Ensemble Member)")
print("=" * 60)

DISEASE_CONFIG = {
    "seed": 42,
    "img_size": 640,            
    "batch_size": 16,
    "epochs": 80,
    "patience": 15,
    "model_variant": "yolov8x.pt",  
    # Disease classes 
    "class_names": ["Caries", "Deep Caries", "Impacted", "Periapical Lesion"],
    "num_classes": 4,
   
    "data_source": {
        "name": "quadrant_enumeration_disease",
        "base_path": "/kaggle/input/datasets/truthisneverlinear/dentex-challenge-2023/training_data/training_data/quadrant-enumeration-disease",
        "json_file": "train_quadrant_enumeration_disease.json",
        "img_subdir": "xrays",
    },
    "output_path": "/kaggle/working/yolo_disease_detection",
    "val_split": 0.2,
    "remove_borders": True,
    "apply_clahe": True,
}

# Class name → index mapping 
DISEASE_CLASS_MAP = {name: idx for idx, name in enumerate(sorted(DISEASE_CONFIG["class_names"]))}
# Sorted: Caries=0, Deep Caries=1, Impacted=2, Periapical Lesion=3
print(f"Disease class map (sorted): {DISEASE_CLASS_MAP}")

# --------------------------------------------------
# D1. Load Disease Annotations
# --------------------------------------------------
source = DISEASE_CONFIG["data_source"]
json_path = os.path.join(source["base_path"], source["json_file"])

if not os.path.exists(json_path):
    print(f"ERROR: Disease annotation not found: {json_path}")
    print("Skipping Part 2 disease YOLO training.")
else:
    with open(json_path, "r") as f:
        disease_data = json.load(f)

    img_dir_disease = os.path.join(source["base_path"], source["img_subdir"])
    id_to_info_disease = {img["id"]: img for img in disease_data["images"]}
    id_to_filename_disease = {img["id"]: img["file_name"] for img in disease_data["images"]}

    # Disease categories from category_id_3
    disease_cat_map = {}
    if "categories_3" in disease_data:
        disease_cat_map = {cat["id"]: cat["name"] for cat in disease_data["categories_3"]}
    print(f"Disease categories: {disease_cat_map}")

    # Group annotations by image (keep disease bboxes only)
    disease_anns_by_img = {}
    skipped_anns = 0
    for ann in disease_data["annotations"]:
        img_id = ann["image_id"]
        disease_id = ann.get("category_id_3")
        if disease_id is None or disease_id not in disease_cat_map:
            skipped_anns += 1
            continue
        disease_name = disease_cat_map[disease_id]
        class_idx = DISEASE_CLASS_MAP.get(disease_name, None)
        if class_idx is None:
            skipped_anns += 1
            continue
        bbox = ann.get("bbox")
        if bbox is None:
            continue
        x, y, w, h = bbox
        if w < 5 or h < 5:
            continue
        if img_id not in disease_anns_by_img:
            disease_anns_by_img[img_id] = []
        disease_anns_by_img[img_id].append({"class_idx": class_idx, "bbox": bbox})

    print(f"Images with disease annotations: {len(disease_anns_by_img)}")
    print(f"Skipped annotations (no disease label): {skipped_anns}")

    # Distribution
    from collections import Counter as _Counter
    dist = _Counter()
    for anns in disease_anns_by_img.values():
        for ann in anns:
            dist[ann["class_idx"]] += 1
    print("Disease annotation distribution:")
    for cls_idx, cls_name in enumerate(sorted(DISEASE_CONFIG["class_names"])):
        print(f"  {cls_name}: {dist.get(cls_idx, 0)}")

    # --------------------------------------------------
    # D2. Create YOLO Disease Dataset
    # --------------------------------------------------
    for split in ["train", "val"]:
        os.makedirs(os.path.join(DISEASE_CONFIG["output_path"], "images", split), exist_ok=True)
        os.makedirs(os.path.join(DISEASE_CONFIG["output_path"], "labels", split), exist_ok=True)

    # Train/val split by image
    disease_img_ids = list(disease_anns_by_img.keys())
    random.shuffle(disease_img_ids)
    n_val = int(len(disease_img_ids) * DISEASE_CONFIG["val_split"])
    val_disease_ids = set(disease_img_ids[:n_val])
    train_disease_ids = set(disease_img_ids[n_val:])
    print(f"\nDisease dataset split: {len(train_disease_ids)} train | {len(val_disease_ids)} val images")

    total_disease_labels = 0
    for split, img_id_set in [("train", train_disease_ids), ("val", val_disease_ids)]:
        split_labels = 0
        for img_id in img_id_set:
            filename = id_to_filename_disease.get(img_id)
            if filename is None:
                continue
            img_path = os.path.join(img_dir_disease, filename)
            if not os.path.exists(img_path):
                continue

            # Preprocess (reuse functions from Part 1)
            processed_img, offset = preprocess_image(
                img_path,
                remove_borders=DISEASE_CONFIG["remove_borders"],
                use_clahe=DISEASE_CONFIG["apply_clahe"],
            )
            if processed_img is None:
                continue

            proc_h, proc_w = processed_img.shape[:2]

            # Save preprocessed image
            dst_img = os.path.join(DISEASE_CONFIG["output_path"], "images", split, filename)
            cv2.imwrite(dst_img, processed_img)

            # Write YOLO labels
            label_path = os.path.join(
                DISEASE_CONFIG["output_path"], "labels", split,
                os.path.splitext(filename)[0] + ".txt"
            )
            with open(label_path, "w") as f:
                for ann in disease_anns_by_img[img_id]:
                    result = coco_to_yolo(ann["bbox"], proc_w, proc_h, offset)
                    if result is None:
                        continue
                    x_c, y_c, w_n, h_n = result
                    f.write(f"{ann['class_idx']} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
                    split_labels += 1

        print(f"  {split}: {split_labels} disease annotations written")
        total_disease_labels += split_labels

    print(f"Total disease labels written: {total_disease_labels}")

    # --------------------------------------------------
    # D3. data.yaml for Disease YOLO
    # --------------------------------------------------
    disease_yaml_content = f"""# DentalScan Disease Detection Dataset (DENTEX 2023 Winner — 2nd Ensemble)
# Auto-generated by phase0_5_yolo_tooth_detection.py Part 2

path: {os.path.abspath(DISEASE_CONFIG['output_path'])}
train: images/train
val: images/val

nc: {DISEASE_CONFIG['num_classes']}
names: {sorted(DISEASE_CONFIG['class_names'])}
"""
    disease_yaml_path = os.path.join(DISEASE_CONFIG["output_path"], "data.yaml")
    with open(disease_yaml_path, "w") as f:
        f.write(disease_yaml_content)
    print(f"\nDisease data.yaml saved: {disease_yaml_path}")

    # --------------------------------------------------
    # D4. Train YOLOv8m for Disease Detection
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("Training YOLOv8m Disease Detector")
    print("=" * 60)

    disease_model = YOLO(DISEASE_CONFIG["model_variant"])

    disease_results = disease_model.train(
        data=disease_yaml_path,
        epochs=DISEASE_CONFIG["epochs"],
        imgsz=DISEASE_CONFIG["img_size"],
        batch=DISEASE_CONFIG["batch_size"],
        patience=DISEASE_CONFIG["patience"],
        save=True,
        save_period=-1,   
        project=DISEASE_CONFIG["output_path"],
        name="train",
        exist_ok=True,
     
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.15,
        degrees=3.0,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.3,       
        mixup=0.0,
        copy_paste=0.0,
        # Detection tuning for small disease regions
        box=7.5,
        cls=1.5,          
        iou=0.5,
        max_det=32,      
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        device=0 if os.path.exists("/dev/nvidia0") or os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        verbose=True,
    )

    # Delete disease last.pt to free disk space
    disease_last_pt = os.path.join(DISEASE_CONFIG["output_path"], "train", "weights", "last.pt")
    if os.path.exists(disease_last_pt):
        os.remove(disease_last_pt)
        print("Deleted disease last.pt (keeping best.pt only)")

    # --------------------------------------------------
    # D5. Evaluate Disease YOLO
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("Evaluating Disease Detector")
    print("=" * 60)

    disease_best_path = os.path.join(DISEASE_CONFIG["output_path"], "train", "weights", "best.pt")
    if not os.path.exists(disease_best_path):
        for root, dirs, files in os.walk(DISEASE_CONFIG["output_path"]):
            for fname in files:
                if fname == "best.pt":
                    disease_best_path = os.path.join(root, fname)
                    break

    disease_best_model = YOLO(disease_best_path)
    disease_val = disease_best_model.val(data=disease_yaml_path, imgsz=DISEASE_CONFIG["img_size"], verbose=True)

    print("\n" + "=" * 40)
    print("DISEASE YOLO VALIDATION METRICS")
    print("=" * 40)
    print(f"  AP@50:      {disease_val.box.map50:.4f}  (target: 0.60+)")
    print(f"  AP@75:      {disease_val.box.map75:.4f}  (target: 0.35+)")
    print(f"  AP@50-95:   {disease_val.box.map:.4f}   (target: 0.30+)")
    print(f"  Precision:  {disease_val.box.mp:.4f}")
    print(f"  Recall:     {disease_val.box.mr:.4f}")
    print(f"  F1 (est):   {2 * disease_val.box.mp * disease_val.box.mr / (disease_val.box.mp + disease_val.box.mr + 1e-8):.4f}")
    print("\n  Per-class AP@50:")
    class_names = DISEASE_CONFIG["class_names"]
    for i, cls_name in enumerate(class_names):
        if i < len(disease_val.box.ap50):
            print(f"    {cls_name:<22} {disease_val.box.ap50[i]:.4f}")


    # --------------------------------------------------
    # D6. Copy Disease YOLO to Working Directory
    # --------------------------------------------------
    final_disease_path = "/kaggle/working/yolo_disease_best.pt"
    shutil.copy2(disease_best_path, final_disease_path)
    print(f"\nDisease YOLO model copied to: {final_disease_path}")

    print("\n" + "=" * 60)
    print("Phase 0.5 COMPLETE - Both models trained:")
    print("=" * 60)
    print(f"  Tooth detector:   {final_model_path}")
    print(f"  Disease detector: {final_disease_path}")

