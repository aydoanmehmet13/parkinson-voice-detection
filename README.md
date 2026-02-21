# 🎙️ ParkinsonVoiceNet

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-88.9%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> A deep learning-powered web application that detects Parkinson's disease from voice recordings using a custom 1D Convolutional Neural Network.

---

## 📋 Overview

ParkinsonVoiceNet is an end-to-end machine learning pipeline that analyzes raw voice recordings to detect Parkinson's disease. The project spans the full lifecycle of a data science project — from exploratory data analysis and acoustic feature engineering to deep learning model development and deployment as an interactive web application.

The system extracts 142-dimensional acoustic feature vectors (MFCC, ZCR, Mel-Spectrogram) from `.wav` files and feeds them into a custom 1D CNN architecture trained on 540 voice recordings, achieving **88.9% accuracy** and an **F1 score of 0.897**.

> ⚠️ **Disclaimer:** This application is intended for research purposes only and should not be used for clinical diagnosis.

---

## 🖥️ Web Application Preview
<p align="center">
<img src="assets/Streamlit.png" alt="ParkinsonVoiceNet Streamlit UI" width="600">
</p>

## 🎯 Key Highlights

- 🔊 **End-to-End Audio Pipeline:** Raw `.wav` files → acoustic features → model prediction
- 🧠 **Custom 1D CNN Architecture:** Designed and trained from scratch using PyTorch
- 📊 **Comprehensive Model Comparison:** CNN vs XGBoost vs Random Forest benchmarking
- 🌐 **Interactive Web Application:** Real-time voice upload and prediction via Streamlit
- 🔬 **Rich Visualizations:** Mel-Spectrogram, MFCC heatmaps, waveform and bandpass analysis
- 📈 **Balanced Dataset:** 276 Parkinson / 264 Healthy — near-perfect class balance

---

## 📊 Model Performance

| Model | Accuracy | F1 Score | Notes |
|---|---|---|---|
| **1D CNN** | **0.889** | **0.897** | Best performer |
| XGBoost | 0.591 | 0.600 | Baseline |
| Random Forest | 0.589 | 0.600 | Baseline |

The 1D CNN significantly outperforms classical ML models because convolutional layers capture local temporal patterns within the feature vector — relationships that tree-based models cannot model effectively.

**CNN Classification Report:**

| Class | Precision | Recall | F1 Score |
|---|---|---|---|
| Healthy (0) | 0.76 | 0.87 | 0.81 |
| Parkinson (1) | 0.89 | 0.79 | 0.83 |
| **Weighted Avg** | **0.83** | **0.82** | **0.82** |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Audio Processing | librosa, SciPy |
| Deep Learning | PyTorch |
| Classical ML | scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web Interface | Streamlit |
| Development | Jupyter Notebook, VS Code |

---

## 🧠 Model Architecture
```
Input (142) → Conv1D(32, k=3) → ReLU → MaxPool(2)
           → Conv1D(64, k=3) → ReLU → MaxPool(2)
           → Flatten → Linear(128) → Dropout(0.3)
           → Linear(1) → Sigmoid → Output
```

- **Conv1D layers** detect local patterns across the 142-feature acoustic vector
- **MaxPool** reduces dimensionality while preserving dominant features
- **Dropout(0.3)** regularizes the network to prevent overfitting
- **BCEWithLogitsLoss** for binary classification, **Adam** optimizer (lr=0.001)
- **30 epochs**, batch size 32 — final training loss: ~0.17

---

## 🔬 Feature Engineering

Each voice recording is processed into a **142-dimensional feature vector**:

| Feature | Dimensions | Description |
|---|---|---|
| MFCC | 13 | Mel-Frequency Cepstral Coefficients — captures vocal timbre |
| ZCR | 1 | Zero-Crossing Rate — measures signal noisiness and tremor |
| Mel-Spectrogram | 128 | Frequency-time energy representation |
| **Total** | **142** | |

All features are standardized using `StandardScaler` before model training and inference.

---

## 📂 Dataset

**[Movement Disorders Voice Dataset](https://www.kaggle.com/datasets/cycoool29/movement-disorders-voice)** — Kaggle

- 540 voice recordings (276 Parkinson, 264 Healthy)
- Near-balanced class distribution — no class weighting required
- `.wav` format — sampled at 3 seconds per recording for consistency

Additionally, EDA was performed on the **UCI Oxford Parkinson's Dataset** (195 records, 22 biomedical features) to understand the clinical feature space before transitioning to raw audio.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation
```bash
# Clone the repository
git clone https://github.com/aydoanmehmet13/parkinson-voice-detection.git
cd parkinson-voice-detection

# Install dependencies
pip install -r requirements.txt

# Launch the web application
streamlit run app.py
```

### Retraining the Model

> To retrain from scratch, download the dataset from Kaggle and update the data paths in `ParkinsonVoiceNet.ipynb` (Cells 19 and 36) to match your local directory structure.
```bash
# Launch notebook
jupyter notebook ParkinsonVoiceNet.ipynb
```

---

## 📁 Project Structure
```
parkinson-voice-detection/
├── ParkinsonVoiceNet.ipynb   # Full pipeline: EDA → features → models → CNN
├── app.py                    # Streamlit web application
├── cnn.pth                   # Trained CNN weights
├── scaler.pkl                # Fitted StandardScaler for inference
├── parkinson.csv             # UCI Parkinson's dataset
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🗺️ Project Roadmap

The project was developed across 4 phases:

| Phase | Description |
|---|---|
| **Phase 1 — EDA** | UCI dataset exploration, correlation analysis, class distribution |
| **Phase 2 — Baseline Models** | Logistic Regression, Random Forest, XGBoost on UCI features |
| **Phase 3 — Audio & Deep Learning** | librosa feature extraction, 1D CNN, model comparison |
| **Phase 4 — Deployment** | Streamlit web app with real-time prediction |

---

## 🔮 Future Improvements

- [ ] 2D CNN on Mel-Spectrogram images for richer spatial feature learning
- [ ] LSTM / Transformer for sequential audio modeling
- [ ] Multi-class diagnosis (Parkinson, Alzheimer, Healthy)
- [ ] Streamlit Cloud deployment for public access
- [ ] Larger dataset integration for improved generalization
- [ ] REST API for integration with clinical tools

---

## 🎓 Learning Outcomes

Through this project, I gained hands-on experience in:

- **Audio Signal Processing** — librosa feature extraction, bandpass filtering, spectrogram analysis
- **Deep Learning from Scratch** — designing, training, and evaluating a PyTorch CNN
- **End-to-End ML Pipeline** — from raw data to a deployed web application
- **Model Evaluation** — accuracy, F1, confusion matrix, classification report
- **Software Engineering** — modular code, version control, reproducible experiments

---

## 👤 Developer

**Mehmet Aydoğan**

🎓 Electrical & Electronics Engineering Student @ İzmir Democracy University  
🔗 LinkedIn: [linkedin.com/in/mehmet-aydoganEE](https://linkedin.com/in/mehmet-aydoganEE)  
📧 Email: aydoanmehmet13@gmail.com  
💻 GitHub: [@aydoanmehmet13](https://github.com/aydoanmehmet13)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

⭐ If you found this project useful, consider giving it a star!

*Built with 🎙️ audio, 🧠 deep learning, and ☕ coffee.*