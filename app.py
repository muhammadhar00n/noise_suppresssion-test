from flask import Flask, request, render_template, send_file
import os
from process_audio import process_audio_in_chunks
import soundfile as sf

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Route for home page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded!"
        
        file = request.files['file']
        if file.filename == '':
            return "No file selected!"
        
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Process the uploaded file
        cleaned_audio = process_audio_in_chunks(file_path, 'TFLiteModel.tflite')
        output_path = os.path.join(OUTPUT_FOLDER, f"cleaned_{file.filename}")
        sf.write(output_path, cleaned_audio, 16000)

        return send_file(output_path, as_attachment=True)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


