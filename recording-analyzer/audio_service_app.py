from flask import Flask, request, jsonify
import librosa
import numpy as np

app = Flask(__name__)

def identify_notes_from_audio_file(file_path, hop_length=1024, threshold=5):
    # Load the audio file
    
    y, sr = librosa.load(file_path, sr=None)  # sr=None preserves the original sample rate

    # Calculate the short-time Fourier transform (STFT)
    stft = np.abs(librosa.stft(y, hop_length=hop_length))

    # Compute the chroma feature



    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)

    # Convert the chroma feature into note names
    notes = []
    for i in range(chroma.shape[1]):
        note_index = np.argmax(chroma[:, i])
        note_name = librosa.midi_to_note(note_index + 60)  # Convert to note name
        notes.append(note_name)

    # Group similar consecutive notes and reduce noise
    smoothed_notes = []
    current_note = notes[0]
    current_count = 1

    for note in notes[1:]:
        if note == current_note:
            current_count += 1
        else:
            if current_count >= threshold:  # Only keep the note if it lasts long enough
                smoothed_notes.append(current_note)
            current_note = note
            current_count = 1

    # Add the last note if it lasted long enough
    if current_count >= threshold:
        smoothed_notes.append(current_note)
    
    return smoothed_notes

@app.route('/api/process', methods=['POST'])
def process_audio():
    # Save the received file
    
    file_path = "/shared-data/output.wav"
    # Identify notes from the recorded audio
    print(file_path)
    notes = identify_notes_from_audio_file(file_path)
    print(notes)
    
    # Return the identified notes as a response
    return jsonify({"notes": notes})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
