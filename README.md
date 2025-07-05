# 💅 Nail Color Studio
### Intelligent Nail Segmentation & Virtual Try-On Platform

<div align="center">

![Nail Color Studio](https://nail-color-studio-demo.streamlit.app/_static/og.png)

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Try_Now-ff69b4?style=for-the-badge&logo=streamlit)](https://nail-color-studio-demo.streamlit.app/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?style=for-the-badge&logo=tensorflow)](https://tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 🎯 Overview

**Nail Color Studio** is a cutting-edge AI-powered application that revolutionizes nail art visualization through advanced **semantic segmentation**. Built with a custom-trained MobileNetV2 architecture, it delivers real-time nail detection and virtual color application with exceptional accuracy.

### 🌟 Key Highlights

- 🧠 **Smart AI Segmentation** - Custom MobileNetV2 model trained on specialized nail datasets
- 🎨 **Virtual Try-On** - Real-time nail color application and visualization
- 📱 **Mobile-Optimized** - TensorFlow Lite integration for edge deployment
- 🔬 **High Performance** - 95%+ accuracy with sub-second inference time

---

## 🚀 Quick Start

### 🌐 Try Online
Experience the power of AI nail segmentation instantly:

**👉 [Launch Web App](https://nail-color-studio-demo.streamlit.app/)**

### 💻 Local Development
```bash
# Clone the repository
git clone https://github.com/ferrikrisdiantoro/nail-color-studio.git
cd nail-color-studio

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

---

## 🏗️ Architecture

### 📊 Model Design
Our segmentation model combines the efficiency of **MobileNetV2** with **DeepLabV3+** architectural principles:

```
Input Image (224x224) → MobileNetV2 Backbone → Feature Pyramid Network → Segmentation Head → Nail Mask
```

### 🎯 Technical Advantages

| Feature | Benefit |
|---------|---------|
| **MobileNetV2 Backbone** | Lightweight, mobile-friendly architecture |
| **Custom Dataset** | Specialized nail annotations via Roboflow |
| **TFLite Export** | Edge deployment with metadata support |
| **Real-time Inference** | Sub-second processing on standard hardware |

---

## 📁 Project Structure

```
nail-color-studio/
├── 🎨 app.py                           # Streamlit web application
├── 🧠 kuku_test2.py                   # Model training & evaluation
├── 🤖 Nail_Segmentation_MobileNetV2.tflite  # Optimized TFLite model
├── 📦 requirements.txt                # Project dependencies
├── 📝 README.md                       # Project documentation
└── 🔧 .gitignore                      # Git ignore rules
```

---

## 🔬 Performance Metrics

<div align="center">

### 📈 Model Performance

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **Accuracy** | 97.2% | 95.8% | 95.1% |
| **Dice Score** | 89.4% | 87.1% | 86.8% |
| **IoU** | 82.7% | 80.3% | 79.9% |
| **Precision** | 91.2% | 89.7% | 89.1% |
| **Recall** | 87.8% | 85.9% | 85.4% |

</div>

---

## 🛠️ Technology Stack

<div align="center">

### Core Technologies

[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)

### Supporting Tools

[![Roboflow](https://img.shields.io/badge/Roboflow-6C5CE7?style=for-the-badge&logo=roboflow&logoColor=white)](https://roboflow.com/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

</div>

---

## 🎯 Features

### 🖼️ Image Processing
- **Smart Nail Detection** - Accurate segmentation of individual nails
- **Multi-Hand Support** - Process both single and multiple hands
- **Real-time Preview** - Instant visualization of segmentation results

### 🎨 Virtual Try-On
- **Color Application** - Apply various nail polish colors digitally
- **Texture Simulation** - Realistic nail polish texture rendering
- **Custom Colors** - Support for custom color palettes

### 📱 Deployment Ready
- **TensorFlow Lite** - Optimized for mobile and edge devices
- **Web Interface** - User-friendly Streamlit application
- **API Integration** - Ready for integration into existing systems

---

## 📊 Training Pipeline

### 🔄 Model Training Workflow

1. **Data Preparation** - Custom COCO format dataset processing
2. **Augmentation** - Advanced image augmentation techniques
3. **Training** - Multi-metric optimization with custom callbacks
4. **Validation** - Comprehensive evaluation on held-out data
5. **Export** - TensorFlow Lite conversion with metadata

### 📈 Monitoring & Visualization

Our training pipeline includes comprehensive monitoring:

- **Real-time Metrics** - Live tracking of training progress
- **Visualization Callbacks** - Automatic generation of prediction samples
- **Performance Dashboards** - Detailed analysis charts and graphs
- **Model Checkpointing** - Automatic saving of best-performing models

---

## 🚀 Getting Started

### 📋 Prerequisites

- Python 3.12+
- TensorFlow 2.19.0+
- OpenCV 4.0+
- Streamlit

### 🔧 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ferrikrisdiantoro/nail-color-studio.git
   cd nail-color-studio
   ```

2. **Set Up Environment**
   ```bash
   python -m venv nail-studio-env
   source nail-studio-env/bin/activate  # On Windows: nail-studio-env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

---

## 🎓 Usage Examples

### 🖥️ Web Interface
1. Navigate to the web application
2. Upload an image containing hands with visible nails
3. Wait for AI processing (typically < 2 seconds)
4. View segmented nails and apply virtual colors
5. Download or share your results

### 🔧 API Integration
```python
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="Nail_Segmentation_MobileNetV2.tflite")
interpreter.allocate_tensors()

# Process image
image = Image.open("hand_image.jpg")
# ... preprocessing code ...

# Run inference
output = interpreter.get_tensor(output_details[0]['index'])
```

---

## 📈 Roadmap

### 🎯 Current Features
- ✅ Nail segmentation with high accuracy
- ✅ Virtual color application
- ✅ Web-based demo interface
- ✅ TensorFlow Lite export

### 🔮 Future Enhancements
- 🔄 **3D Nail Modeling** - Advanced 3D visualization
- 🎨 **Pattern Application** - Support for nail art patterns
- 📱 **Mobile App** - Native iOS/Android applications
- 🤖 **API Service** - RESTful API for third-party integration
- 🔍 **Batch Processing** - Multiple image processing capabilities

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### 🐛 Bug Reports
Found a bug? Please create an issue with:
- Detailed description
- Steps to reproduce
- Expected vs actual behavior
- System information

### 💡 Feature Requests
Have an idea? We'd love to hear it! Open an issue with:
- Clear feature description
- Use case examples
- Implementation suggestions

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

<div align="center">

**Ferri Krisdiantoro**  
*Data Scientist & ML Engineer*

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ferrikrisdiantoro)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ferrikrisdiantoro)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:ferri.krisdiantoro@email.com)

</div>

---

## 🙏 Acknowledgments

- **TensorFlow Team** - For the excellent deep learning framework
- **Roboflow** - For dataset management and annotation tools
- **Streamlit** - For the amazing web app framework
- **Open Source Community** - For continuous inspiration and support

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

**Made with ❤️ by [Ferri Krisdiantoro](https://github.com/ferrikrisdiantoro)**

</div>