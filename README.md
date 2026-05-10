# Dental Disease Detection

🌐 **Live Web App:** [https://shreyaslingwal.github.io/Dental_disease_detection/phase4_webapp/](https://shreyaslingwal.github.io/Dental_disease_detection/phase4_webapp/)

An end-to-end multi-label dental diagnostic ensemble pipeline that detects teeth, classifies dental diseases (Caries, Deep Caries, Impacted, Periapical Lesions), visualizes clinical findings using GradCAM, and generates professional clinical reports using a fine-tuned LLaMA 3 Large Language Model. The project utilizes the DENTEX 2023 dataset structure.

##  Project Overview

The AI diagnostic pipeline is split into logical phases to optimize computational resources and ensure clinical-grade accuracy:

1. **Phase 0.5 - Tooth Detection:** A YOLOv8x model trained to accurately detect and localize teeth (incorporating FDI numbering mappings).
2. **Phase 1 - Disease Detection & Classification:** A dual-model approach utilizing a Swin Transformer (Swin-S) and YOLOX to classify dental pathologies.
3. **Phase 2 - GradCAM Bridge:** Generates explainability heatmaps using GradCAM, overlaying bounding boxes and visual evidence to allow clinical verification of AI predictions.
4. **Phase 3 - Clinical Report Generation:** A fine-tuned LLaMA 3 model (via LoRA adapter) takes the localized disease findings and outputs a natural language, professional clinical diagnostic report.
5. **Phase 4 - Deployment:** A FastAPI server (hosted on Google Colab, tunneled via ngrok) serving a modern web application for an interactive, horizontally scrollable dashboard.

##  DENTEX 2023 Dataset Overview

The DENTEX dataset comprises panoramic dental X-rays obtained from three different institutions using standard clinical conditions but varying equipment and imaging protocols, resulting in diverse image quality reflecting heterogeneous clinical practice. The dataset includes X-rays from patients aged 12 and above, randomly selected from the hospital's database to ensure patient privacy and confidentiality.

To enable effective use of the FDI system, the dataset is hierarchically organized into three types of data:

- **(a)** 693 X-rays labeled for quadrant detection and quadrant classes only.
- **(b)** 634 X-rays labeled for tooth detection with quadrant and tooth enumeration classes.
- **(c)** 1005 X-rays fully labeled for abnormal tooth detection with quadrant, tooth enumeration, and diagnosis classes.

The diagnosis class includes four specific categories: **caries, deep caries, periapical lesions, and impacted teeth**. An additional challenge identified and addressed during the training of the models is the inherent class imbalance present within these pathology categories across the 3,529 total disease tooth crops (from 678 unique images):
## Distribution of Tooth Level Diseases labels in the dataset:
- **Caries:** 2,189 samples
- **Impacted:** 604 samples
- **Deep Caries:** 578 samples
- **Periapical Lesion:** 158 samples

## 📊 Performance Results

The ensemble models were rigorously evaluated on the dataset. Below are the key performance metrics across the different architectures:

### YOLOv8x Tooth Detector

| Metric | Value |
| :--- | :--- |
| **mAP@50** | 0.930 |
| **Precision** | 0.911 |
| **Recall** | 0.985 |

### Swin Transformer (Disease Classification)

| Metric | Value |
| :--- | :--- |
| **Validation Accuracy** | 0.8419 |
| **Validation mAP** | 0.8244 |
| **Overall Accuracy** | 0.84 |
| **Macro Avg Precision** | 0.77 |
| **Macro Avg Recall** | 0.79 |
| **Macro Avg F1-score** | 0.77 |
| **Weighted Avg Precision** | 0.84 |
| **Weighted Avg Recall** | 0.84 |
| **Weighted Avg F1-score** | 0.84 |

**Class-Specific Metrics (Swin Transformer):**

| Pathology | Precision | Recall | F1-score |
| :--- | :--- | :--- | :--- |
| **Caries** | 0.88 | 0.90 | 0.89 |
| **Deep Caries** | 0.68 | 0.54 | 0.60 |
| **Impacted** | 0.91 | 0.97 | 0.94 |
| **Periapical Lesion** | 0.60 | 0.75 | 0.67 |

### Disease YOLOX

| Metric | Value |
| :--- | :--- |
| **AP@50** | 0.8564 |
| **AP@75** | 0.8092 |
| **AP@50-95** | 0.6758 |
| **Precision (Box)** | 0.904 |
| **Recall (Box)** | 0.821 |
| **F1 Score** | 0.8607 |

**Class-Specific Metrics (YOLOX AP@50):**

| Pathology | AP@50 |
| :--- | :--- |
| **Caries** | 0.8231 |
| **Deep Caries** | 0.8638 |
| **Impacted** | 0.9736 |
| **Periapical Lesion** | 0.7653 |

## ⚙️ How to Use the Live App

Because the AI models require heavy GPU computation, the backend is hosted dynamically via Google Colab. To run the full application:

1. **Start the AI Server:**
   - Open `phase4_colab_server.py` in Google Colab (with a GPU runtime like T4).
   - Run the notebook to load the YOLO, Swin-S, and LLaMA 3 models into memory.
   - The final cell will generate a public **ngrok URL**.
2. **Connect the Web App:**
   - Open the [Live Web App](https://shreyaslingwal.github.io/Dental_disease_detection/phase4_webapp/).
   - Paste your generated ngrok URL into the "Connect" input box at the top right of the dashboard.
   - Upload a dental radiograph and generate a clinical report!
