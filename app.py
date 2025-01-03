from flask import Flask, request, render_template, send_file, jsonify, Response
from werkzeug.utils import secure_filename
import os
import threading
import subprocess
from main_processing_script import process_video_with_dashboard
from camera_script import generate_camera_frames

app = Flask(__name__)
app.config['TEMP_FOLDER'] = 'temp'
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

progress = {'percentage': 0}
camera_process = None  # Global variable to manage the live camera subprocess


def process_video(upload_path, output_video_path, report_path):
    global progress
    try:
        progress['percentage'] = 0  # Reset progress
        process_video_with_dashboard(upload_path, output_video_path)  # Process video
        # Generate processed video and report
        generated_report_path = process_video_with_dashboard(upload_path, output_video_path)
        if generated_report_path != report_path:
            os.rename(generated_report_path, report_path)  # Ensure the report file is accessible
        progress['percentage'] = 100  # Processing complete
    except Exception as e:
        progress['percentage'] = -1  # Indicate error
        print(f"Error during processing: {e}")


@app.route('/')
def home():
    return render_template('home.html', title="Emotion Vista")


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['TEMP_FOLDER'], filename)
        output_video_path = os.path.join(app.config['TEMP_FOLDER'], f"processed_{filename}.mp4")
        report_path = os.path.join(app.config['TEMP_FOLDER'], f"report_{filename}.txt")

        # Remove existing files if they exist
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        if os.path.exists(report_path):
            os.remove(report_path)
            
        file.save(upload_path)

        # Process video in a separate thread
        threading.Thread(target=process_video, args=(upload_path, output_video_path, report_path)).start()
        return jsonify({'message': 'Processing started', 'filename': filename})


@app.route('/progress', methods=['GET'])
def get_progress():
    return jsonify(progress)

@app.route('/live_camera_feed')
def live_camera_feed():
    return Response(
        generate_camera_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_process
    if camera_process is None or camera_process.poll() is not None:  # Not running
        camera_process = subprocess.Popen(['python', 'camera_script.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return jsonify({'message': 'Live camera started'})
    else:
        return jsonify({'error': 'Camera is already running'}), 400

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_process
    if camera_process and camera_process.poll() is None:  # Running
        camera_process.terminate()
        camera_process = None
        return jsonify({'message': 'Live camera stopped'})
    else:
        return jsonify({'error': 'No live camera to stop'}), 400


if __name__ == "__main__":
    app.run(debug=True)
