import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import pickle

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)



st.set_page_config(
    page_title="Parkinson Ses Analizi",
    page_icon="🎙️",
    layout="centered"
)


class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 35, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


model = CNN1D()
model.load_state_dict(torch.load("cnn.pth", map_location="cpu"))
model.eval()


def extract_features(file):
    y, sr = librosa.load(file, sr=None, duration=3)
    y = y.astype(float)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)
    return np.concatenate([mfcc, [zcr], mel])


def predict(features):
    features_scaled = scaler.transform([features])
    tensor = torch.FloatTensor(features_scaled[0]).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()
    return prob


st.markdown(
    """
    <h1 style='text-align: center; color: #e5e7eb;'>🎙️ Parkinson Ses Analizi</h1>
    <p style='text-align: center; font-size: 16px; color: #9ca3af;'>
    Derin öğrenme tabanlı CNN modeli ile ses analizi
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("### 📂 Ses Dosyası Yükleme")

uploaded_file = st.file_uploader(
    "Lütfen .wav formatında bir ses dosyası seçin",
    type=["wav"]
)

if uploaded_file is not None:
    features = extract_features(uploaded_file)
    prob = predict(features)

    st.markdown("---")
    st.markdown("### 📊 Model Sonucu")

    if prob > 0.5:
        st.error(f"🧠 **Parkinson olasılığı yüksek**\n\n**%{prob*100:.2f}**")
    else:
        st.success(f"✅ **Parkinson olasılığı düşük**\n\n**%{(1-prob)*100:.2f}**")


st.markdown(
    """
    <style>
    .stApp {
        background-color: #0f172a;
        color: #e5e7eb;
    }

    .block-container {
        background-color: #020617;
        padding: 2.5rem;
        border-radius: 14px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        max-width: 720px;
        margin-top: 2rem;
    }

    label, .stMarkdown {
        color: #e5e7eb !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.caption("⚠️ Bu uygulama yalnızca araştırma amaçlıdır. Klinik teşhis için kullanılmamalıdır. Model %88.9 doğruluk oranıyla çalışmaktadır.")