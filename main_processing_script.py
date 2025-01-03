import cv2
import numpy as np
import librosa
import joblib
from keras.models import load_model
from moviepy.editor import VideoFileClip
import tempfile
import os

# Load the trained audio and face emotion models and the label encoder
audio_model = load_model('models/speech_emotion_recognition.h5')
face_model = load_model('models/face_model_100epochs.h5')
label_encoder = joblib.load('models/label_encoder.pkl')

# Define emotions and initialize trends
emotions = ['anger', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_trends = {emotion: [] for emotion in emotions}

# Load Haar cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
faceDetect = cv2.CascadeClassifier(cascade_path)

# Face emotion labels
face_labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Function to extract features from an audio chunk
def extract_features(audio_chunk, rate):
    mfccs = librosa.feature.mfcc(y=audio_chunk, sr=rate, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

# Function to predict emotion from audio
def predict_audio_emotion(audio_chunk, rate):
    features = extract_features(audio_chunk, rate)
    features = np.reshape(features, (1, -1, 1))
    prediction = audio_model.predict(features)
    label = label_encoder.inverse_transform([np.argmax(prediction)])
    return label[0], prediction[0]

# Function to generate the dashboard with waveforms for audio emotions
def draw_dashboard(frame, emotion_probs, emotion_trends):
    dashboard_width = 800
    dashboard_height = frame.shape[0]
    dashboard = np.ones((dashboard_height, dashboard_width, 3), dtype=np.uint8) * 255

    bar_height = dashboard_height // 8
    bar_width = 300
    wave_width = 300
    vertical_padding = 10

    for i, emotion in enumerate(emotions):
        cv2.putText(dashboard, emotion, (10, (i + 1) * bar_height - vertical_padding), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.rectangle(dashboard, (110, i * bar_height), 
                      (110 + int(bar_width * emotion_probs[i]), (i + 1) * bar_height - vertical_padding), 
                      (0, 255, 0), -1)

        wave_x = 450
        wave_y_start = i * bar_height + vertical_padding
        wave_y_end = (i + 1) * bar_height - vertical_padding
        wave_y_center = (wave_y_start + wave_y_end) // 2

        cv2.rectangle(dashboard, (wave_x, i * bar_height), (wave_x + wave_width, (i + 1) * bar_height - vertical_padding), (0, 0, 0), 2)

        if len(emotion_trends[emotion]) > 1:
            norm_trend = (np.array(emotion_trends[emotion]) - np.min(emotion_trends[emotion])) / (np.max(emotion_trends[emotion]) - np.min(emotion_trends[emotion]) + 1e-6)
            norm_trend = (norm_trend * (wave_y_end - wave_y_start) // 2).astype(int)

            for j in range(1, len(norm_trend)):
                cv2.line(dashboard, (wave_x + j - 1, wave_y_center - norm_trend[j - 1]), 
                         (wave_x + j, wave_y_center - norm_trend[j]), (0, 0, 0), 2)

    combined_frame = np.hstack((frame, dashboard))
    return combined_frame

# Function to detect face emotion and update counts
# Update function to detect face emotion and return the number of faces detected
def detect_face_emotion(frame, face_emotion_counts):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    faces_detected = 0  # Track the number of detected faces
    for x, y, w, h in faces:
        faces_detected += 1
        sub_face_img = gray[y:y + h, x:x + w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        face_result = face_model.predict(reshaped)
        face_label = np.argmax(face_result, axis=1)[0]

        face_emotion = face_labels_dict[face_label].lower()
        if face_emotion in face_emotion_counts:
            face_emotion_counts[face_emotion] += 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, face_labels_dict[face_label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame, faces_detected

# Update the processing function to use the number of detected faces
def process_video_with_dashboard(video_path, output_path, box_width=500, box_height=500):
    clip = VideoFileClip(video_path)
    audio = clip.audio

    cached_emotion_probs = np.zeros(len(emotions))
    emotion_trends = {emotion: [] for emotion in emotions}

    audio_emotion_counts = {emotion: 0 for emotion in emotions}
    face_emotion_counts = {emotion: 0 for emotion in emotions}
    total_faces_detected = 0  # Track the total number of faces detected

    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    video_writer = None

    for t, frame in clip.iter_frames(with_times=True, fps=clip.fps):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_resized = cv2.resize(frame, (box_width, box_height))

        audio_chunk = audio.subclip(t, t + (1 / clip.fps)).to_soundarray()

        if len(audio_chunk) > 0:
            audio_chunk = audio_chunk.flatten()
            emotion_label, emotion_probs = predict_audio_emotion(audio_chunk, audio.fps)
            if emotion_label in audio_emotion_counts:
                audio_emotion_counts[emotion_label] += 1
            cached_emotion_probs[:] = emotion_probs

            for i, emotion in enumerate(emotions):
                emotion_trends[emotion].append(emotion_probs[i])
                if len(emotion_trends[emotion]) > 300:
                    emotion_trends[emotion].pop(0)

        frame_with_face_emotion, faces_detected = detect_face_emotion(frame_resized, face_emotion_counts)
        total_faces_detected += faces_detected  # Increment total faces detected
        combined_frame = draw_dashboard(frame_with_face_emotion, cached_emotion_probs, emotion_trends)

        if video_writer is None:
            height, width, _ = combined_frame.shape
            video_writer = cv2.VideoWriter(temp_video.name, cv2.VideoWriter_fourcc(*'mp4v'), clip.fps, (width, height))

        video_writer.write(combined_frame)

    video_writer.release()
    final_clip = VideoFileClip(temp_video.name).set_audio(audio)
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

# Generate emotion analysis report
    report_path = os.path.splitext(output_path)[0] + "_report.txt"
    with open(report_path, 'w') as report_file:
        report_file.write("Audio Emotion Analysis (%):\n")
        total_audio_emotions = sum(audio_emotion_counts.values())
        for emotion in emotions:
            percentage = (audio_emotion_counts[emotion] / total_audio_emotions) * 100 if total_audio_emotions > 0 else 0
            report_file.write(f"{emotion}: {percentage:.2f}%\n")

        report_file.write("\nFace Emotion Analysis (%):\n")
        for emotion in emotions:
            percentage = (face_emotion_counts[emotion] / total_faces_detected) * 100 if total_faces_detected > 0 else 0
            report_file.write(f"{emotion}: {percentage:.2f}%\n")
    return report_path
