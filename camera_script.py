import cv2
import numpy as np
import librosa
import joblib
import pyaudio
import threading
from keras.models import load_model
import signal
import sys

# Load models and configurations
audio_model = load_model('models/speech_emotion_recognition.h5')
face_model = load_model('models/face_model_100epochs.h5')
label_encoder = joblib.load('models/label_encoder.pkl')

emotions = ['anger', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_trends = {emotion: [] for emotion in emotions}

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
faceDetect = cv2.CascadeClassifier(cascade_path)

# Audio configurations
p = pyaudio.PyAudio()
chunk_size = 1024
sample_format = pyaudio.paInt16
channels = 1
rate = 16000

# Graceful shutdown
def shutdown_handler(signum, frame):
    print("Shutting down camera script...")
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

# Feature extraction for audio
def extract_features(audio_chunk, rate):
    mfccs = librosa.feature.mfcc(y=audio_chunk, sr=rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Predict emotion from audio
def predict_audio_emotion(audio_chunk, rate):
    features = extract_features(audio_chunk, rate)
    features = np.reshape(features, (1, -1, 1))
    prediction = audio_model.predict(features)
    label = label_encoder.inverse_transform([np.argmax(prediction)])
    return label[0], prediction[0]

# Draw the audio dashboard
def draw_audio_dashboard(emotion_probs, emotion_trends):
    dashboard_width = 800
    dashboard_height = 500
    dashboard = np.ones((dashboard_height, dashboard_width, 3), dtype=np.uint8) * 255

    bar_height = dashboard_height // 8
    bar_width = 300
    wave_width = 300
    vertical_padding = 10

    for i, emotion in enumerate(emotions):
        # Labels
        cv2.putText(dashboard, emotion, (10, (i + 1) * bar_height - vertical_padding), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Bars
        cv2.rectangle(dashboard, (110, i * bar_height), 
                      (110 + int(bar_width * emotion_probs[i]), (i + 1) * bar_height - vertical_padding), 
                      (0, 255, 0), -1)

        # Waveforms
        wave_x = 450
        wave_y_start = i * bar_height + vertical_padding
        wave_y_end = (i + 1) * bar_height - vertical_padding
        wave_y_center = (wave_y_start + wave_y_end) // 2

        cv2.rectangle(dashboard, (wave_x, i * bar_height), 
                      (wave_x + wave_width, (i + 1) * bar_height - vertical_padding), (0, 0, 0), 2)

        if len(emotion_trends[emotion]) > 1:
            norm_trend = (np.array(emotion_trends[emotion]) - np.min(emotion_trends[emotion])) / \
                         (np.max(emotion_trends[emotion]) - np.min(emotion_trends[emotion]) + 1e-6)
            norm_trend = (norm_trend * (wave_y_end - wave_y_start) // 2).astype(int)

            for j in range(1, len(norm_trend)):
                cv2.line(dashboard, (wave_x + j - 1, wave_y_center - norm_trend[j - 1]), 
                         (wave_x + j, wave_y_center - norm_trend[j]), (0, 0, 0), 2)

    return dashboard

# Detect face emotion
def detect_face_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    for x, y, w, h in faces:
        sub_face_img = gray[y:y + h, x:x + w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))
        face_result = face_model.predict(reshaped)
        face_label = np.argmax(face_result, axis=1)[0]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotions[face_label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame

# Audio streaming in real time
def audio_stream():
    stream = p.open(format=sample_format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk_size)
    while True:
        audio_data = np.frombuffer(stream.read(chunk_size), dtype=np.int16).astype(np.float32)
        audio_data /= np.max(np.abs(audio_data))  # Normalize
        audio_label, audio_probs = predict_audio_emotion(audio_data, rate)

        print(f"Audio Emotion: {audio_label}")

        for i, emotion in enumerate(emotions):
            emotion_trends[emotion].append(audio_probs[i])
            if len(emotion_trends[emotion]) > 300:
                emotion_trends[emotion].pop(0)

        yield audio_probs

# Generate frames with face emotions and audio dashboard
def generate_camera_frames():
    cap = cv2.VideoCapture(0)
    audio_thread = threading.Thread(target=audio_stream)
    audio_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        audio_probs = next(audio_stream())  # Fetch audio probabilities
        frame_resized = cv2.resize(frame, (500, 500))
        frame_with_face = detect_face_emotion(frame_resized)
        dashboard = draw_audio_dashboard(audio_probs, emotion_trends)
        combined_frame = np.hstack((frame_with_face, dashboard))

        _, buffer = cv2.imencode('.jpg', combined_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    p.terminate()
