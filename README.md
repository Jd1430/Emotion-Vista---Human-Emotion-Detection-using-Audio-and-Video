![Web Output4](https://github.com/user-attachments/assets/240a8129-8bcd-40ac-8058-68acbe43c8f6)# Emotion Vista

## Approach
Emotion Vista is a web application designed for emotion detection from videos and live camera feeds using machine learning models. The platform combines both **audio emotion recognition** and **face emotion detection** to process multimedia content, generate reports, and display real-time emotional analysis.

### Key Components:
1. **Video Upload**: Users can upload videos for processing. The system processes both face emotions and audio emotions.
2. **Live Camera Feed**: The application allows users to analyze emotions from a live video feed from a webcam.
3. **Real-time Progress Tracking**: A progress bar is displayed while videos are being processed.
4. **Downloadable Reports**: Once the video is processed, users can download the processed video and a detailed report of the emotional analysis.

## Technologies Used
- **Frontend**: HTML, CSS, JavaScript (with AJAX for asynchronous updates)
- **Backend**: Python, Flask
- **Machine Learning**:
  - **Audio Emotion Recognition**: `librosa`, Keras model
  - **Face Emotion Detection**: OpenCV, Keras model
- **Camera Feed**: OpenCV for webcam integration
- **Threading**: For asynchronous video processing
- **Subprocess**: For managing live camera feed processes

## Features
- **Upload Video**: Upload videos for emotion detection from both faces and audio.
- **Live Camera Feed**: Start and stop a live camera feed for real-time emotion analysis.
- **Progress Bar**: Track the progress of video processing.
- **Emotion Dashboard**: Display emotion charts for both audio and visual data.
- **Download Processed Files**: Once processing is complete, download the processed video and a detailed emotion report.
- **Responsive UI**: Optimized for both desktop and mobile devices.

## Instructions to Set Up and Run the Project

### Prerequisites:
1. **Python 3.7 or higher**
2. **pip** package manager for installing dependencies
3. **A webcam** for live camera functionality

### Steps to Set Up the Project:

1. **Clone the Repository**  
   Clone the repository to your local machine:
   ```bash
   git clone <repository-url>
   cd Emotion_Vista
2. **Install Dependencies**  
   Install the required libraries using the requirements.txt file:
   ```bash
   pip install -r requirements.txt

3. **Download Pretrained Models**  
   The project uses pretrained models for both audio emotion recognition and face emotion detection. Place the following models in the models/ directory:

    - speech_emotion_recognition.h5 (audio model)**
    - face_model_100epochs.h5 (face emotion model)**
    - label_encoder.pkl (label encoder for audio emotions)**
   
4. **Run the Flask Application**  
   Start the Flask application by running the following command:
   ```bash
   python app.py

5. **Access the Application**  
   Open a web browser and navigate to http://127.0.0.1:5000.

## Steps to Use the Project:

### 1. **Upload Video**:
- Click on **"Process the Video"** and upload a video file for emotion analysis.
- Wait for the progress bar to fill up as the video is being processed.

### 2. **Real-time Camera Feed**:
- Click **"Open Live Camera"** to start the live camera feed and see real-time face emotion detection.
- You can stop the live camera feed anytime by clicking **"Stop Camera"**.

### 3. **Download Processed Files**:
- Once the processing is complete, the **"Download Processed Video"** and **"Download Report"** buttons will become visible.
- Click to download the processed video and the generated report.

## Snapshots:

### 1. **Home Page**:
![Web Output1](https://github.com/user-attachments/assets/1d417f8f-76e7-4174-988a-013821300a0f)

### 2. **Progress Bar (While Processing Video)**:
![Web Output2](https://github.com/user-attachments/assets/07540c76-4a79-4664-b019-21b56a3b2e86)

### 3. **Live Camera Feed**:
![Web Output4](https://github.com/user-attachments/assets/85bbceee-9563-4d6a-88b8-c3a7bbc5352c)






