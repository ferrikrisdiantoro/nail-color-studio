# 💅 Nail Color Studio

**AI-Powered Nail Segmentation & Visualization using TensorFlow, TFLite & Streamlit**

[![Streamlit Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-green)](https://nail-color-studio-demo.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

**Nail Color Studio** is a computer vision-based web application that automatically detects and segments fingernails from hand images and lets users try out virtual nail polish. Built with a custom segmentation model trained using TensorFlow & MobileNetV2, and deployed with a Streamlit front-end powered by a TFLite-optimized model.

### 🔗 [Live Demo →](https://nail-color-studio-demo.streamlit.app/)

---

## 🧠 Model Highlights

- ✨ **Architecture**: Custom lightweight segmentation model based on `MobileNetV2` backbone (DeepLabV3+ inspired).
- 📱 **TFLite Conversion**: Optimized for mobile inference with metadata via `tflite-support`.
- 🎯 **Performance Metrics**:
  - Dice Coefficient
  - IoU Score
  - Precision & Recall

---

## 🗂 Project Structure

```
nail-color-studio/
│
├── notebook/
│   └── kuku_test2.ipynb            # Jupyter notebook for experimentation
├── kuku_test2.py                   # Full model training pipeline
├── Nail_Segmentation_MobileNetV2.tflite  # Final optimized TFLite model
├── app.py                          # Streamlit web app
├── requirements.txt                # Python dependencies
└── .gitignore
```

---

## 🏗 How It Works

### 🔬 1. Model Development (`kuku_test2.py`)
- Uses the **Roboflow API** to download annotated COCO-style datasets.
- Trains a **custom U-Net-style model** with a MobileNetV2 encoder.
- Includes:
  - Training visualization callback
  - Custom metrics & losses (Dice, IoU, Precision, Recall)
  - TFLite conversion & metadata population

### 🌐 2. Streamlit App (`app.py`)
- Loads the `TFLite` model using `tflite-runtime`.
- Accepts image input and runs segmentation on the client side.
- Allows users to apply custom **nail polish colors** interactively.

---

## 🚀 Getting Started

### ⚙️ 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 🧪 2. Run Streamlit App Locally

```bash
streamlit run app.py
```

> Make sure `Nail_Segmentation_MobileNetV2.tflite` is present in the root directory.

---

## 🧰 Tech Stack

| Area             | Tech / Tool |
|------------------|-------------|
| Model Training   | TensorFlow, MobileNetV2, COCO Dataset, Roboflow |
| Optimization     | TFLite, MetadataWriter |
| Frontend         | Streamlit |
| Visualization    | Matplotlib |
| Deployment       | Streamlit Cloud |
| Dataset Format   | COCO JSON (via Roboflow) |

---

## 📊 Example Results

| Input Image | Ground Truth Mask | Predicted Mask |
|-------------|-------------------|----------------|
| ![original](assets/sample1.jpg) | ![gt](assets/sample1_gt.jpg) | ![pred](assets/sample1_pred.jpg) |

*(For illustration. Real results shown in the demo.)*

---

## 🔬 Model Training Insights

The training pipeline supports advanced metrics logging and automatic performance monitoring, including:
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint
- Live training curve plotting

> To retrain the model, run:

```bash
python kuku_test2.py
```

---

## 📦 Exported Model

The model is saved as:

```bash
Nail_Segmentation_MobileNetV2.tflite
```

Includes:
- Embedded label metadata
- Input normalization info
- Ready for TensorFlow Lite Task Library or Streamlit inference

---

## 🧪 Requirements

- Python ≥ 3.9
- TensorFlow ≥ 2.10
- Streamlit
- Roboflow SDK
- TFLite-Support

---

## 📸 Screenshot

![Demo UI](assets/demo_screenshot.png)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

- [Roboflow](https://roboflow.com/) for dataset hosting
- [TensorFlow Lite](https://www.tensorflow.org/lite) for model deployment tools
- [Streamlit](https://streamlit.io) for easy deployment

---

## ✍️ Author

**Ferri Krisdiantoro**  
[GitHub](https://github.com/ferrikrisdiantoro) | [LinkedIn](https://linkedin.com/in/ferrikrisdiantoro)
