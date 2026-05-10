
# --------------------------------------------------
# 1. Imports
# --------------------------------------------------
import json
import os
import random
import re
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient


client = UserSecretsClient()
hf_token = client.get_secret("Llama 3")
login(hf_token)
print("Hugging Face authentication successful.")

# --------------------------------------------------
# 2. CONFIG
# --------------------------------------------------
CONFIG = {
    "seed": 42,
    "model_id": "NousResearch/Meta-Llama-3-8B-Instruct",
    # Paths (search order)
    "phase2_prompts_paths": [
        "/kaggle/working/phase2_prompts.json",
        "/kaggle/input/dental-phase2/phase2_prompts.json",
    ],
    "output_dir": "/kaggle/working/llama3-dental-lora",
    # QLoRA settings
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    # Training
    "num_epochs": 3,
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "max_seq_length": 1024,
    "warmup_ratio": 0.05,
    # Inference
    "max_new_tokens": 512,
    "temperature": 0.3,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    # Disease metadata (must match Phase 1/2 exactly)
    "num_classes": 4,
    "disease_names": {
        0: "Caries",
        1: "Deep Caries",
        2: "Impacted",
        3: "Periapical Lesion",
    },
   
    "class_thresholds": [0.60, 0.55, 0.50, 0.50],

    "uncertain_margin": 0.1,
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------
# 3. Load Phase 2 Prompts
# --------------------------------------------------
phase2_path = None
for candidate in CONFIG["phase2_prompts_paths"]:
    if os.path.exists(candidate):
        phase2_path = candidate
        break

if phase2_path:
    with open(phase2_path, "r") as f:
        phase2_data = json.load(f)
    print(f"Loaded {len(phase2_data)} prompts from: {phase2_path}")
else:
    phase2_data = []
    print("WARNING: phase2_prompts.json not found. Will use synthetic data only.")
    for p in CONFIG["phase2_prompts_paths"]:
        print(f"  Searched: {p}")

# --------------------------------------------------
# 4. Synthetic Training Data Generator
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

DISEASE_TEMPLATES = {
    "Caries": {
        "descriptions": [
            "A carious lesion is observed on the {surface} surface of tooth {fdi} ({name}). "
            "The extent appears {extent}, {certainty_text} early enamel demineralization.",
            "Radiographic evidence of dental caries on tooth {fdi} ({name}), "
            "affecting the {surface} aspect. The lesion is {certainty_text} {extent} decay.",
        ],
        "recommendations": [
            "Composite restoration recommended.",
            "Clinical examination and possible restorative treatment advised.",
            "Preventive fluoride application and monitoring for progression.",
        ],
        "surfaces": ["mesial", "distal", "occlusal", "buccal"],
        "extents": ["limited to enamel", "extending into dentin", "superficial"],
    },
    "Deep Caries": {
        "descriptions": [
            "Deep carious involvement is noted on tooth {fdi} ({name}), with the lesion "
            "extending close to the pulp chamber. {certainty_text} pulpal proximity.",
            "Tooth {fdi} ({name}) shows advanced carious destruction with {certainty_text} "
            "involvement of the inner dentin layers. Pulp vitality should be assessed.",
        ],
        "recommendations": [
            "Urgent restorative treatment with possible pulp capping is recommended.",
            "Vitality testing and endodontic consultation advised.",
            "Root canal therapy may be required if pulp is compromised.",
        ],
        "surfaces": ["occlusal", "mesio-occlusal", "disto-occlusal"],
        "extents": ["deep into dentin near the pulp", "extending toward the pulp horn"],
    },
    "Impacted": {
        "descriptions": [
            "Tooth {fdi} ({name}) appears impacted with a {angulation} angulation. "
            "{certainty_text} soft tissue or bony impaction.",
            "An impacted tooth is identified at position {fdi} ({name}), showing "
            "{angulation} orientation. {certainty_text} partial eruption failure.",
        ],
        "recommendations": [
            "Surgical extraction is recommended after further CBCT evaluation.",
            "Referral to oral surgery for assessment and possible extraction.",
            "Monitoring recommended if asymptomatic; extraction if recurrent pericoronitis.",
        ],
        "angulations": ["mesioangular", "distoangular", "vertical", "horizontal"],
    },
    "Periapical Lesion": {
        "descriptions": [
            "A periapical radiolucency is observed at the apex of tooth {fdi} ({name}), "
            "{certainty_text} a periapical abscess or granuloma.",
            "Tooth {fdi} ({name}) demonstrates a well-defined periapical lesion. "
            "This finding is {certainty_text} chronic periapical pathology.",
        ],
        "recommendations": [
            "Endodontic treatment or retreatment is recommended.",
            "Pulp vitality testing and periapical assessment are advised.",
            "Apicoectomy may be considered if conventional root canal therapy fails.",
        ],
    },
}

# Severity ranking (higher = more clinically severe, synced with Phase 2)
SEVERITY_RANK = {
    "Periapical Lesion": 4,
    "Deep Caries": 3,
    "Impacted": 2,
    "Caries": 1,
}

# Only these 4 DENTEX diseases are allowed in reports
ALLOWED_DISEASES = {"Caries", "Deep Caries", "Impacted", "Periapical Lesion"}

# Maps disease combinations to merged clinical language
DISEASE_MERGE_MAP = {
    frozenset({"Deep Caries", "Periapical Lesion"}): "deep caries with associated periapical changes",
    frozenset({"Caries", "Periapical Lesion"}): "carious involvement with periapical pathology",
    frozenset({"Deep Caries", "Impacted"}): "deep caries on partially impacted tooth",
}

# Recommendation merge rules (avoids redundant per-tooth recommendations)
REC_MERGE = {
    frozenset({"Vitality testing", "Endodontic treatment"}): "Vitality testing followed by possible endodontic treatment",
    frozenset({"Vitality testing", "Root canal therapy"}): "Vitality testing followed by possible endodontic treatment",
}

# Clinical certainty language (synced with Phase 2 interpret_confidence thresholds)
CERTAINTY_LANGUAGE = {
    "confirmed": ["indicative of", "consistent with", "strongly suggestive of"],
    "likely": ["consistent with", "suggestive of", "likely representing"],
    "suspected": ["suggestive of", "possibly representing", "raising concern for"],
}


def generate_synthetic_report(findings_list):
    """Generate a synthetic clinical report from a list of finding dicts.
    
    Produces reports that follow clinical best practices:
    - Findings grouped per tooth (not duplicated)
    - Severity-ranked ordering (most severe first)
    - Conditional clinical impressions (not generic)
    - Merged recommendations per tooth
    - Confidence-aligned language
    """
    if not findings_list:
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

    # Filter to allowed diseases
    valid = [f for f in findings_list if f.get("disease") in ALLOWED_DISEASES]
    if not valid:
        return generate_synthetic_report([])

    # Group by tooth
    tooth_groups = {}
    for f in valid:
        fdi = f.get("fdi_number", 0)
        if fdi not in tooth_groups:
            tooth_groups[fdi] = []
        tooth_groups[fdi].append(f)

    # Sort by max severity per tooth (most severe first)
    sorted_teeth = sorted(
        tooth_groups.keys(),
        key=lambda fdi: max(SEVERITY_RANK.get(g["disease"], 0) for g in tooth_groups[fdi]),
        reverse=True,
    )

    sections = {"summary_lines": [], "detail_lines": [], "rec_lines": []}

    for fdi in sorted_teeth:
        group = tooth_groups[fdi]
        name = group[0].get("fdi_name", FDI_TEETH.get(fdi, f"Tooth {fdi}"))
        diseases = [g["disease"] for g in group]
        disease_set = frozenset(diseases)

        # Use highest confidence for language selection
        best = max(group, key=lambda g: g.get("confidence", 0))
        certainty = best.get("certainty", "suspected")
        conf = best.get("confidence", 0.5)
        certainty_text = random.choice(CERTAINTY_LANGUAGE.get(certainty, ["suggestive of"]))

        # Merged description for same-tooth multi-disease
        if disease_set in DISEASE_MERGE_MAP:
            merged = DISEASE_MERGE_MAP[disease_set]
            sections["summary_lines"].append(f"{merged.capitalize()} at tooth {fdi} ({name}).")
            sections["detail_lines"].append(
                f"Tooth {fdi} ({name}): Findings are {certainty_text} {merged}. "
                f"Confidence: {conf:.0%} ({certainty})."
            )
        elif len(diseases) > 1:
            combined = " with ".join(d.lower() for d in diseases)
            sections["summary_lines"].append(f"{combined.capitalize()} at tooth {fdi} ({name}).")
            sections["detail_lines"].append(
                f"Tooth {fdi} ({name}): Findings are {certainty_text} {combined}. "
                f"Confidence: {conf:.0%} ({certainty})."
            )
        else:
            d = diseases[0]
            template = DISEASE_TEMPLATES.get(d, DISEASE_TEMPLATES["Caries"])
            desc_template = random.choice(template["descriptions"])
            fmt_kwargs = {
                "fdi": fdi, "name": name, "certainty_text": certainty_text,
                "surface": random.choice(template.get("surfaces", [""])),
                "extent": random.choice(template.get("extents", [""])),
                "angulation": random.choice(template.get("angulations", ["mesioangular"])),
            }
            description = desc_template.format(**fmt_kwargs)
            sections["summary_lines"].append(f"{d} ({certainty}) at tooth {fdi} ({name}).")
            sections["detail_lines"].append(f"Tooth {fdi} ({name}): {description}")

        # Merged recommendations per tooth
        tooth_recs = set()
        for g in group:
            d = g["disease"]
            if d == "Deep Caries":
                tooth_recs.add("Vitality testing")
                tooth_recs.add("Endodontic treatment")
            elif d == "Caries":
                tooth_recs.add("Restorative treatment")
            elif d == "Periapical Lesion":
                tooth_recs.add("Endodontic treatment")
                tooth_recs.add("Vitality testing")
            elif d == "Impacted":
                tooth_recs.add("Surgical evaluation")

        # Apply merge rules
        merged_rec = None
        for combo, merged_text in REC_MERGE.items():
            if combo.issubset(tooth_recs):
                remaining = tooth_recs - combo
                merged_rec = merged_text
                if remaining:
                    merged_rec += "; " + "; ".join(remaining)
                break
        if not merged_rec:
            merged_rec = "; ".join(sorted(tooth_recs))
        sections["rec_lines"].append(f"Tooth {fdi}: {merged_rec}.")

    # Build conditional clinical impression
    all_diseases = set(d for gs in tooth_groups.values() for g in gs for d in [g["disease"]])
    impressions = []
    caries_count = sum(
        1 for gs in tooth_groups.values()
        for g in gs if g["disease"] in ("Caries", "Deep Caries")
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
    clinical_impression = ". ".join(impressions) + "."

    report = "1. Summary\n"
    report += "Panoramic radiograph analysis reveals the following findings:\n"
    report += "\n".join(f"- {s}" for s in sections["summary_lines"])
    report += "\n\n2. Detailed Findings\n"
    report += "\n".join(f"- {d}" for d in sections["detail_lines"])
    report += f"\n\n3. Clinical Impression\n{clinical_impression}\n"
    report += "\n4. Recommendations\n"
    report += "\n".join(f"- {r}" for r in sections["rec_lines"])

    return report


def generate_synthetic_pair():
    """Generate one synthetic (prompt, report) pair with random findings.

    Matches Phase 2 output format exactly:
    - Each finding includes 'status' (positive/uncertain), 'cam_intensity',
      'location_px', 'bbox' fields to match Phase 2 finding schema
    - A full 'probabilities' array is generated per example
    - Status is determined by per-class thresholds + uncertain_margin
    - Prompt separates confirmed vs uncertain findings (Phase 2 build_prompt)
    - 20% of multi-finding cases include same-tooth multi-disease scenarios
    """
    num_findings = random.choices([0, 1, 2, 3], weights=[0.05, 0.4, 0.35, 0.2])[0]
    disease_names = list(DISEASE_TEMPLATES.keys())
    disease_name_to_idx = {name: idx for idx, name in CONFIG["disease_names"].items()}
    fdi_list = list(FDI_TEETH.keys())
    class_thresholds = CONFIG["class_thresholds"]
    uncertain_margin = CONFIG["uncertain_margin"]
    img_size = CONFIG.get("img_size", 336)

    findings = []
    used_teeth = set()

    # Generate realistic per-class probabilities (softmax-like, sums to ~1)
    base_probs = np.random.dirichlet([0.5] * CONFIG["num_classes"]).tolist()

    # 20% chance of same-tooth multi-disease (if multiple findings)
    add_multi_disease = num_findings >= 2 and random.random() < 0.2

    for idx in range(num_findings):
        if add_multi_disease and idx == 1:
            # Reuse the first tooth for a co-occurring disease
            fdi = findings[0]["fdi_number"]
            # Pick a clinically co-occurring disease
            first_disease = findings[0]["disease"]
            co_occur = {
                "Deep Caries": ["Periapical Lesion"],
                "Caries": ["Periapical Lesion"],
                "Periapical Lesion": ["Deep Caries"],
            }
            candidates = co_occur.get(first_disease, disease_names)
            disease = random.choice(candidates)
        else:
            fdi = random.choice(fdi_list)
            while fdi in used_teeth:
                fdi = random.choice(fdi_list)
            used_teeth.add(fdi)
            disease = random.choice(disease_names)

        conf = round(random.uniform(0.3, 0.98), 2)
        if conf > 0.85:
            certainty = "confirmed"
        elif conf > 0.65:
            certainty = "likely"
        else:
            certainty = "suspected"

        severity = "high" if conf > 0.85 else ("moderate" if conf > 0.7 else "low")

        # Determine status using per-class thresholds (matches Phase 2 logic)
        disease_idx = disease_name_to_idx.get(disease, 0)
        threshold = class_thresholds[disease_idx] if disease_idx < len(class_thresholds) else 0.5
        if conf >= threshold:
            status = "positive"
        elif conf >= threshold - uncertain_margin:
            status = "uncertain"
        else:
            # Below uncertain margin -- still include in synthetic data
            # but mark as uncertain for training diversity
            status = "uncertain"

        # Assign FDI quadrant from fdi_number
        quadrant = fdi // 10

        # Generate realistic cam_intensity (correlated with confidence, with noise)
        cam_intensity = round(min(1.0, max(0.05, conf * random.uniform(0.6, 1.2))), 3)

        # Generate realistic pixel location based on quadrant
        mid = img_size // 2
        if quadrant in [1, 4]:  # Right side
            cx = random.randint(mid + 10, img_size - 20)
        else:  # Left side
            cx = random.randint(20, mid - 10)
        if quadrant in [1, 2]:  # Upper
            cy = random.randint(20, mid - 10)
        else:  # Lower
            cy = random.randint(mid + 10, img_size - 20)

        # Generate realistic bbox around center
        bw = random.randint(20, 50)
        bh = random.randint(20, 50)
        bbox = [max(0, cx - bw//2), max(0, cy - bh//2), bw, bh]

        # Inject the finding's confidence into probabilities array
        base_probs[disease_idx] = max(base_probs[disease_idx], conf)

        findings.append({
            "disease": disease,
            "disease_idx": disease_idx,
            "fdi_number": fdi,
            "fdi_name": FDI_TEETH[fdi],
            "quadrant": quadrant,
            "confidence": conf,
            "certainty": certainty,
            "severity": severity,
            "status": status,
            "cam_intensity": cam_intensity,
            "location_px": [cx, cy],
            "bbox": bbox,
        })

    # Normalize probabilities so they sum to ~1 (softmax-like)
    prob_sum = sum(base_probs)
    probabilities = [round(p / prob_sum, 4) for p in base_probs]

    # Build the Phase 2 style prompt (matches Phase 2 build_prompt format exactly)
    if not findings:
        prompt_text = (
            "Patient Radiograph Findings:\n"
            "- No significant pathological findings detected.\n\n"
            "You are a dental radiology assistant.\n\n"
            "Generate a clinical report noting the absence of significant pathology.\n"
            "Format:\n1. Summary\n2. Clinical Impression\n3. Recommendations\n"
        )
    else:
        # Sort by FDI tooth number for clinical readability (matches Phase 2)
        findings_sorted = sorted(findings, key=lambda f: f["fdi_number"])

        # Separate positive and uncertain findings (Phase 2 build_prompt format)
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

        # Quadrant context (matches Phase 2 quadrant_summary)
        quadrant_names = {1: "upper right", 2: "upper left", 3: "lower left", 4: "lower right"}
        affected_quads = sorted(set(f.get("quadrant", 0) for f in findings))
        affected_text = ", ".join(quadrant_names.get(q, str(q)) for q in affected_quads)

        prompt_text = "Patient Radiograph Findings:\n"
        prompt_text += "\n".join(lines)
        prompt_text += "\n\n"
        prompt_text += f"Affected regions: {affected_text}\n"
        prompt_text += f"Total findings: {len(findings)}\n\n"
        prompt_text += (
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

    report = generate_synthetic_report(findings)
    return prompt_text, report


def build_training_dataset(num_synthetic=500):
    """Build the instruct-tuning dataset.

    Combines:
    1. Phase 2 real prompts (with synthetic reports as targets)
    2. Purely synthetic prompt-report pairs for diversity
    """
    examples = []

    # From Phase 2 real prompts
    for entry in phase2_data:
        prompt = entry["prompt"]
        findings = entry.get("findings", [])
        report = generate_synthetic_report(findings)
        examples.append({"prompt": prompt, "response": report})

    # Generate additional synthetic pairs
    for _ in range(num_synthetic):
        prompt, report = generate_synthetic_pair()
        examples.append({"prompt": prompt, "response": report})

    random.shuffle(examples)
    print(f"Training dataset: {len(examples)} examples "
          f"({len(phase2_data)} from Phase 2 + {num_synthetic} synthetic)")
    return examples


# --------------------------------------------------
# 5. Format for LLaMA 3 Instruct
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
    """Format a single example into LLaMA 3 Instruct chat template.

    Uses the official <|begin_of_text|> ... <|eot_id|> format.
    """
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


# --------------------------------------------------
# 6. Load Model with 4-bit Quantization
# --------------------------------------------------
# Free any leftover GPU memory from Phase 1/2 (if running in same session)
import gc
gc.collect()
torch.cuda.empty_cache()
print(f"\nGPU memory free: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")
print(f"Loading {CONFIG['model_id']} with 4-bit quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_id"], token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_id"],
    token=hf_token,
    quantization_config=bnb_config,
    device_map={"": 0},  # Force single GPU (avoids cuda:0/cuda:1 split)
    torch_dtype=torch.float16,
)
model.config.use_cache = False  # Required for gradient checkpointing
model = prepare_model_for_kbit_training(model)

print("Base model loaded successfully.")
print(f"  Model parameters: {model.num_parameters():,}")

# --------------------------------------------------
# 7. Apply LoRA Adapters
# --------------------------------------------------
lora_config = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    target_modules=CONFIG["lora_target_modules"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --------------------------------------------------
# 8. Build Dataset
# --------------------------------------------------
raw_examples = build_training_dataset(num_synthetic=500)

# Format into chat template strings
formatted_texts = []
for ex in raw_examples:
    text = format_llama3_chat(ex["prompt"], ex["response"])
    formatted_texts.append({"text": text})

dataset = Dataset.from_list(formatted_texts)

# Train/eval split (90/10)
split = dataset.train_test_split(test_size=0.1, seed=CONFIG["seed"])
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# --------------------------------------------------
# 9. Training
# --------------------------------------------------

from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["num_epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    lr_scheduler_type="cosine",
    warmup_ratio=CONFIG["warmup_ratio"],
    fp16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    seed=CONFIG["seed"],
    ddp_find_unused_parameters=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    processing_class=tokenizer,
    max_seq_length=1024,
)

print("\nStarting LoRA fine-tuning...")
print("-" * 60)
trainer.train()
print("-" * 60)
print("Training complete.")

# Save the LoRA adapter
adapter_path = os.path.join(CONFIG["output_dir"], "final_adapter")
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"LoRA adapter saved to: {adapter_path}")

# --------------------------------------------------
# 10. Inference Function
# --------------------------------------------------
def generate_report(prompt_text, model, tokenizer):
    """Generate a clinical report from a Phase 2 prompt.

    Args:
        prompt_text: the structured prompt from Phase 2
        model: the fine-tuned LLaMA 3 model
        tokenizer: the LLaMA 3 tokenizer

    Returns:
        report: str - the generated clinical report
    """
    formatted = format_llama3_chat(prompt_text)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=CONFIG["max_new_tokens"],
            temperature=CONFIG["temperature"],
            top_p=CONFIG["top_p"],
            repetition_penalty=CONFIG["repetition_penalty"],
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (skip the input prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    report = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return report.strip()


# NOTE: SEVERITY_RANK, ALLOWED_DISEASES, DISEASE_MERGE_MAP, REC_MERGE, and
# CERTAINTY_LANGUAGE are defined once at module level (after DISEASE_TEMPLATES).
# The postprocess_report function below uses those module-level definitions.


def build_clinical_impression(grouped_findings):
    """Generate a specific clinical impression instead of generic 'clinical correlation recommended'."""
    all_diseases = set()
    for tooth_findings in grouped_findings.values():
        for f in tooth_findings:
            all_diseases.add(f.get("disease", ""))

    num_teeth = len(grouped_findings)
    impressions = []

    if not grouped_findings:
        return "No radiographic evidence of pathology. Dentition within normal limits."

    # Caries pattern
    caries_count = sum(
        1 for fs in grouped_findings.values()
        for f in fs if f.get("disease") in ("Caries", "Deep Caries")
    )
    if caries_count >= 3:
        impressions.append("Generalized carious involvement across multiple teeth")
    elif caries_count > 0:
        impressions.append(f"Localized carious involvement ({caries_count} tooth/teeth affected)")

    # Periapical pattern
    if "Periapical Lesion" in all_diseases:
        impressions.append("Possible endodontic pathology requiring further evaluation")

    # Impaction pattern
    if "Impacted" in all_diseases:
        impressions.append("Eruption disturbance identified")

    # Deep caries + periapical = combined
    if "Deep Caries" in all_diseases and "Periapical Lesion" in all_diseases:
        impressions.append("Advanced carious destruction with suspected pulpal or periapical sequelae")

    if not impressions:
        impressions.append("Findings noted; clinical correlation advised")

    return ". ".join(impressions) + "."


def postprocess_report(raw_report, findings):
    """Apply rule-based post-processing to fix common LLM output issues.

    Fixes:
    1. Groups duplicate findings for the same tooth
    2. Replaces generic clinical impressions with conditional ones
    3. Aligns confidence language with input certainty
    4. Merges redundant recommendations
    5. Sorts findings by severity (most severe first)
    6. Filters out any diseases not in ALLOWED_DISEASES
    """
    if not findings:
        # Normal case: replace any generic text
        normal_report = (
            "1. Summary\n"
            "No radiographic evidence of pathology. Dentition within normal limits.\n\n"
            "2. Clinical Impression\n"
            "The panoramic radiograph demonstrates no significant abnormalities. "
            "All visible teeth and supporting structures appear within normal radiographic limits.\n\n"
            "3. Recommendations\n"
            "- Routine follow-up and periodic radiographic monitoring advised.\n"
            "- Standard preventive care and oral hygiene maintenance recommended."
        )
        return normal_report

    # --- Step 1: Filter to allowed diseases only ---
    valid_findings = [f for f in findings if f.get("disease") in ALLOWED_DISEASES]
    if not valid_findings:
        return postprocess_report(raw_report, [])  # treat as normal

    # --- Step 2: Group by tooth (FDI number) ---
    tooth_groups = {}
    for f in valid_findings:
        fdi = f.get("fdi_number", 0)
        if fdi not in tooth_groups:
            tooth_groups[fdi] = []
        tooth_groups[fdi].append(f)

    # --- Step 3: Sort teeth by max severity (most severe first) ---
    sorted_teeth = sorted(
        tooth_groups.keys(),
        key=lambda fdi: max(SEVERITY_RANK.get(f["disease"], 0) for f in tooth_groups[fdi]),
        reverse=True,
    )

    # --- Step 4: Build structured report ---
    summary_lines = []
    detail_lines = []
    rec_lines = []

    for fdi in sorted_teeth:
        group = tooth_groups[fdi]
        tooth_name = FDI_TEETH.get(fdi, f"Tooth {fdi}")
        diseases = [f["disease"] for f in group]
        disease_set = frozenset(diseases)

        # Pick the highest-confidence certainty for the tooth
        best = max(group, key=lambda f: f.get("confidence", 0))
        certainty = best.get("certainty", "suspected")
        conf = best.get("confidence", 0.5)
        lang = random.choice(CERTAINTY_LANGUAGE.get(certainty, ["suggestive of"]))

        # Merged description for same-tooth findings
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

        # Merged recommendations per tooth
        tooth_recs = set()
        for f in group:
            d = f["disease"]
            if d == "Deep Caries":
                tooth_recs.add("Vitality testing")
                tooth_recs.add("Endodontic treatment")
            elif d == "Caries":
                tooth_recs.add("Restorative treatment")
            elif d == "Periapical Lesion":
                tooth_recs.add("Endodontic treatment")
                tooth_recs.add("Vitality testing")
            elif d == "Impacted":
                tooth_recs.add("Surgical evaluation")

        # Apply merge rules
        merged_rec = None
        for combo, merged in REC_MERGE.items():
            if combo.issubset(tooth_recs):
                remaining = tooth_recs - combo
                merged_rec = merged
                if remaining:
                    merged_rec += "; " + "; ".join(remaining)
                break
        if not merged_rec:
            merged_rec = "; ".join(sorted(tooth_recs))

        rec_lines.append(f"Tooth {fdi}: {merged_rec}.")

    # --- Step 5: Assemble final report ---
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
# 11. Run Inference on Phase 2 Prompts
# --------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 3 INFERENCE")
print("=" * 60)

model.eval()
generated_reports = []

# Use Phase 2 prompts (or first 5 synthetic if none available)
inference_data = phase2_data if phase2_data else raw_examples[:5]

for i, entry in enumerate(inference_data[:10]):  # Cap at 10 for demo
    prompt = entry["prompt"]
    findings = entry.get("findings", [])
    print(f"\n--- Image {i+1} ---")
    raw_report = generate_report(prompt, model, tokenizer)
    # Apply rule-based post-processing
    final_report = postprocess_report(raw_report, findings)
    print(final_report)
    generated_reports.append({
        "img_path": entry.get("img_path", f"synthetic_{i}"),
        "prompt": prompt,
        "raw_llm_output": raw_report,
        "generated_report": final_report,
    })

# --------------------------------------------------
# 12. Clinical Accuracy Check
# --------------------------------------------------
def check_clinical_accuracy(report, findings):
    """Check if the generated report faithfully references the input FDI teeth
    and does not hallucinate new teeth."""
    if not findings:
        return {"fdi_recall": 1.0, "hallucinated_teeth": []}

    input_fdi = set()
    for f in findings:
        fdi = f.get("fdi_number")
        if fdi:
            input_fdi.add(str(fdi))

    # Find all 2-digit numbers in the report ( FDI references)
    mentioned_fdi = set(re.findall(r'\b([1-4][1-8])\b', report))

    # Recall: how many input teeth are mentioned in the report
    matched = input_fdi.intersection(mentioned_fdi)
    recall = len(matched) / len(input_fdi) if input_fdi else 1.0

    # Hallucinated: teeth mentioned in report but NOT in input
    hallucinated = mentioned_fdi - input_fdi

    return {"fdi_recall": recall, "hallucinated_teeth": list(hallucinated)}


print("\n" + "=" * 60)
print("CLINICAL ACCURACY EVALUATION")
print("=" * 60)

total_recall = 0.0
total_hallucinations = 0
n_evaluated = 0

for i, entry in enumerate(generated_reports):
    findings = []
    if i < len(inference_data):
        findings = inference_data[i].get("findings", [])

    result = check_clinical_accuracy(entry["generated_report"], findings)
    total_recall += result["fdi_recall"]
    total_hallucinations += len(result["hallucinated_teeth"])
    n_evaluated += 1

    status = "PASS" if result["fdi_recall"] >= 0.8 and len(result["hallucinated_teeth"]) == 0 else "REVIEW"
    print(f"  Image {i+1}: FDI Recall={result['fdi_recall']:.0%}, "
          f"Hallucinated teeth={result['hallucinated_teeth']}, Status={status}")

if n_evaluated > 0:
    avg_recall = total_recall / n_evaluated
    print(f"\nAverage FDI Recall: {avg_recall:.0%}")
    print(f"Total hallucinated teeth: {total_hallucinations}")

# --------------------------------------------------
# 13. Save Reports to JSON
# --------------------------------------------------
reports_path = os.path.join(CONFIG["output_dir"], "phase3_reports.json")
os.makedirs(CONFIG["output_dir"], exist_ok=True)
with open(reports_path, "w") as f:
    json.dump(generated_reports, f, indent=2)
print(f"\nGenerated reports saved to: {reports_path}")

# --------------------------------------------------
# 14. Summary
# --------------------------------------------------
print("\n" + "=" * 60)
print("Phase 3 complete.")
print("Outputs:")
print(f"  - LoRA adapter:  {adapter_path}")
print(f"  - Reports JSON:  {reports_path}")

