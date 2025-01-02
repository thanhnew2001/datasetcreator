from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import uuid
import zipfile
import time
import threading
import shutil
import json
from werkzeug.utils import secure_filename
from datetime import datetime
from split_transcribe import split_audio, transcribe_audio_files  # Import your existing functions

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Config for file upload
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'wav'}

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Define a route to serve the index.html file
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.wav")
        file.save(file_path)
        
        # Start the audio processing in a separate thread
        threading.Thread(target=process_audio, args=(file_path, file_id)).start()
        
        return jsonify({"message": "File uploaded successfully", "file_id": file_id}), 200
    else:
        return jsonify({"error": "Invalid file format"}), 400

@app.route('/status/<file_id>', methods=['GET'])
def check_status(file_id):
    output_dir = os.path.join(OUTPUT_FOLDER, file_id)
    if os.path.exists(os.path.join(output_dir, 'metadata.csv')):
        zip_path = os.path.join(output_dir, f"{file_id}.zip")
        return jsonify({"status": "ready", "zip_url": f"/download/{file_id}"}), 200
    else:
        return jsonify({"status": "processing"}), 200

@app.route('/download/<file_id>', methods=['GET'])
def download_file(file_id):
    zip_path = os.path.join(OUTPUT_FOLDER, file_id, f"{file_id}.zip")
    if os.path.exists(zip_path):
        return send_from_directory(os.path.dirname(zip_path), os.path.basename(zip_path), as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404

def process_audio(file_path, file_id):
    # 1. Split the audio
    output_dir = os.path.join(OUTPUT_FOLDER, file_id)
    os.makedirs(output_dir, exist_ok=True)

    split_audio(file_path, output_dir)

    # 2. Transcribe the audio
    transcribe_audio_files(output_dir, os.path.join(output_dir, "metadata.csv"))
    
    # 3. Create a ZIP file
    zip_path = os.path.join(output_dir, f"{file_id}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for foldername, subfolders, filenames in os.walk(output_dir):
            for filename in filenames:
                zipf.write(os.path.join(foldername, filename), os.path.relpath(os.path.join(foldername, filename), output_dir))
    
    print(f"ZIP file created at {zip_path}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
