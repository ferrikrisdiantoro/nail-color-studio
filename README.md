# 💅 Nail Color Studio - Intelligent Nail Segmentation and Demo App

![demo](https://nail-color-studio-demo.streamlit.app/_static/og.png)

**Nail Color Studio** is a smart AI-powered application that performs **semantic segmentation on fingernails** using a custom-trained MobileNetV2-based model. This project demonstrates both model development and Streamlit deployment, enabling interactive demos for end-users.

> 🧠 Built by combining deep learning, computer vision, and TensorFlow Lite for real-time experience.

---

## 📌 Live Demo

👉 [Try the Web App on Streamlit](https://nail-color-studio-demo.streamlit.app/)

---

## 📁 Project Structure

```
├── app.py                            # Streamlit frontend app
├── kuku_test2.py                    # Model training, evaluation & TFLite conversion
├── Nail_Segmentation_MobileNetV2.tflite # Exported TFLite model
├── requirements.txt                 # All dependencies
├── .gitignore
```

---

## 🧠 Model Architecture

The custom model is inspired by **DeepLabV3+/FPN-style segmentation**, using **MobileNetV2** as the feature extractor backbone. This makes it:

- ✅ Lightweight for mobile deployment
- ✅ Compatible with TensorFlow Lite and MediaPipe
- ✅ Highly accurate on small objects like fingernails

### ✨ Features

- **Custom COCO Dataset**: Annotated via Roboflow
- **Modular Training Pipeline**: With custom callbacks for Dice, IoU, Precision, Recall
- **Realtime Visualization**: Integrated callbacks to visualize training predictions
- **Export to TFLite**: With metadata support for Task Library inference

---

## 🧪 Model Training

Model training is performed in `kuku_test2.py` or the Colab notebook. It includes:

- Custom dataset loader (`COCO format`)
- Segmentation metrics: `dice`, `IoU`, `precision`, `recall`
- Visualization & analysis tools (matplotlib dashboards)
- TFLite export with embedded label metadata

---

## 🔍 Inference Pipeline

After training, the model is exported to `.tflite` and used in the demo app:

```python
interpreter = tf.lite.Interpreter(model_path="Nail_Segmentation_MobileNetV2.tflite")
```

The demo shows:

- Input nail image
- Predicted mask overlay
- Ability to apply colors or effects

---

## 🚀 Streamlit Demo App

The `app.py` file is a Streamlit-powered interactive app that lets users:

- Upload an image of a hand
- See the segmented nails in real-time
- Apply different nail polish colors digitally

### Run locally

```bash
streamlit run app.py
```

---

## 📦 Installation

Create a clean virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies include:

- `tensorflow==2.19.0`
- `opencv-python-headless`
- `streamlit`
- `tflite-support-nightly`

---

## 🔬 Metrics Visualization

During training, multiple callbacks generate performance plots:

- **Accuracy & Loss**
- **Dice & IoU Coefficients**
- **Precision & Recall**

All charts are saved in high resolution.

---

## 🎯 Performance

| Metric       | Value (Val Set) |
|--------------|------------------|
| Accuracy     | > 0.95           |
| Dice Score   | > 0.87           |
| IoU          | > 0.80           |
| Precision    | > 0.89           |
| Recall       | > 0.85           |

## 🛠️ Tech Stack

- Python 3.12
- TensorFlow
- Streamlit
- OpenCV
- TFLite
- Roboflow (for dataset management)
- Matplotlib

---

## 📌 Author

**Ferri Krisdiantoro**  
Data Scientist & ML Engineer  
[GitHub](https://github.com/ferrikrisdiantoro)

---

## 📃 License

MIT License © 2025 Ferri Krisdiantoro

---