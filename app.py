import streamlit as st
import librosa
import numpy as np
import joblib

# Load saved models
rf = joblib.load("knn_k5_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Feature extraction function (36 features)
def extract_features(file):
    audio, sr = librosa.load(file, sr=22050, duration=5.0, mono=True)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))

    features = np.hstack([
        mfcc_mean,
        chroma_mean,
        centroid,
        bandwidth,
        rolloff,
        zcr
    ])

    return features.reshape(1, -1)

# Streamlit UI
st.title("🎵 Music Instrument Recognition")
st.write("Upload an audio file to identify the musical instrument.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    features = extract_features(uploaded_file)
    features = scaler.transform(features)

    prediction = rf.predict(features)
    instrument = le.inverse_transform(prediction)[0]

    st.success(f"🎶 Predicted Instrument: **{instrument}**")
