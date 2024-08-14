import pyaudio
import numpy as np
import librosa

# Settings for audio capture
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100  # Sample rate
CHUNK = 1024  # Smaller chunk size to reduce overflow risk
BUFFER_SIZE = CHUNK * 4  # Increased buffer size

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Start recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=BUFFER_SIZE)

print("Listening...")

# To store the sequence of notes
note_sequence = []

try:
    while True:
        try:
            # Capture audio chunk
            data = stream.read(CHUNK, exception_on_overflow=False)
            np_data = np.frombuffer(data, dtype=np.float32)

            # Use librosa to calculate frequencies
            fft_result = np.abs(np.fft.fft(np_data))[:CHUNK//2]
            freqs = np.fft.fftfreq(len(fft_result), 1.0/RATE)[:CHUNK//2]

            # Find the peak frequency
            peak_freq = freqs[np.argmax(fft_result)]

            # Ignore very low or zero frequencies
            if peak_freq > 20:  # 20 Hz is the lower bound of human hearing
                # Convert frequency to note
                note = librosa.hz_to_note(peak_freq)
                # Store the note in the sequence
                note_sequence.append(note)
                print(f"Peak Frequency: {peak_freq:.2f} Hz, Note: {note}")
            else:
                print("No significant frequency detected, ignoring...")

        except IOError as e:
            print(f"Input overflowed: {e}")
            continue

except KeyboardInterrupt:
    print("Stopping...")

finally:
    # Stop and close the stream if it's active
    if stream.is_active():
        stream.stop_stream()
    stream.close()
    audio.terminate()

    # Display the sequence of notes played
    if note_sequence:
        print("\nSequence of notes played:")
        print(" -> ".join(note_sequence))
    else:
        print("\nNo valid notes were detected.")
