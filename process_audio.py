import tensorflow as tf
import numpy as np
import soundfile as sf
import librosa

# Load the TFLite model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Function to apply smoothing at overlaps
def apply_smoothing(cleaned_chunks, chunk_size, overlap):
    output = np.zeros((len(cleaned_chunks) * (chunk_size - overlap) + overlap,))
    window = np.hanning(overlap * 2)  # Hanning window to smooth overlaps

    for i, chunk in enumerate(cleaned_chunks):
        chunk = np.squeeze(chunk)  # Flatten the chunk to 1D
        start = i * (chunk_size - overlap)
        end = start + chunk_size
        if i == 0:
            output[start:end] = chunk
        else:
            # Apply smoothing in the overlap region
            output[start:start + overlap] *= window[:overlap]
            output[start:start + overlap] += chunk[:overlap] * window[overlap:]
            output[start + overlap:end] = chunk[overlap:]

    return output


# Process audio in overlapping chunks
def process_audio_in_chunks(file_path, model_path, target_sr=16000, chunk_size=12000, overlap=6000):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=target_sr)
    audio = audio.astype(np.float32)
    
    # Prepare the TFLite interpreter
    interpreter = load_tflite_model(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cleaned_chunks = []

    # Process audio in chunks with overlap
    for start in range(0, len(audio) - chunk_size + 1, chunk_size - overlap):
        chunk = audio[start:start + chunk_size]
        chunk = np.expand_dims(chunk, axis=(0, -1))  # Add batch and channel dimensions
        
        interpreter.set_tensor(input_details[0]['index'], chunk)
        interpreter.invoke()

        # Get the cleaned chunk
        cleaned_chunk = interpreter.get_tensor(output_details[0]['index'])[0]
        cleaned_chunks.append(cleaned_chunk)

    # Apply smoothing to the cleaned chunks
    cleaned_audio = apply_smoothing(cleaned_chunks, chunk_size, overlap)

    return cleaned_audio

    # Save the cleaned audio
    sf.write("cleaned_output.wav", cleaned_audio, target_sr)
    print("Cleaned audio saved as 'cleaned_output.wav'.")

# Run the function
process_audio_in_chunks("noisy_audio.wav", "TFLiteModel.tflite")
