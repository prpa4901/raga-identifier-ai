import pyaudio
import numpy as np
import librosa

# Settings for audio capture
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100  # Sample rate
CHUNK = 1024  # Number of samples per frame

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Start recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Listening...")

try:
    while True:
        # Capture audio chunk
        data = stream.read(CHUNK)
        np_data = np.frombuffer(data, dtype=np.float32)

        # Use librosa to calculate frequencies
        fft_result = np.abs(np.fft.fft(np_data))[:CHUNK//2]
        freqs = np.fft.fftfreq(len(fft_result), 1.0/RATE)[:CHUNK//2]
        
        # Find the peak frequency
        peak_freq = freqs[np.argmax(fft_result)]
        
        # Convert frequency to note
        note = librosa.hz_to_note(peak_freq)
        
        print(f"Peak Frequency: {peak_freq:.2f} Hz, Note: {note}")

except KeyboardInterrupt:
    print("Stopping...")

finally:
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
