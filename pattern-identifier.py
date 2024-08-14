import pyaudio
import numpy as np
import librosa
import scipy.signal as signal

# Settings for audio capture
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100  # Sample rate
CHUNK = 1024  # Smaller chunk size to reduce overflow risk
BUFFER_SIZE = CHUNK * 4  # Increased buffer size

# Frequency range for musical notes
LOW_FREQ = 20    # Lower bound of musical frequencies
HIGH_FREQ = 5000  # Upper bound for most musical instruments

# Minimum energy threshold to filter out noise
ENERGY_THRESHOLD = 0.1

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Start recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=BUFFER_SIZE)

print("Listening...")

# To store the sequence of notes
note_sequence = []

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data)
    return y

try:
    while True:
        try:
            # Capture audio chunk
            data = stream.read(CHUNK, exception_on_overflow=False)
            np_data = np.frombuffer(data, dtype=np.float32)

            # Apply band-pass filter
            filtered_data = apply_bandpass_filter(np_data, LOW_FREQ, HIGH_FREQ, RATE)

            # Use librosa to calculate frequencies
            fft_result = np.abs(np.fft.fft(filtered_data))[:CHUNK//2]
            freqs = np.fft.fftfreq(len(fft_result), 1.0/RATE)[:CHUNK//2]

            # Calculate the energy of the signal
            energy = np.sum(fft_result**2) / len(fft_result)

            # Find the peak frequency only if the energy is above the threshold
            if energy > ENERGY_THRESHOLD:
                peak_freq = freqs[np.argmax(fft_result)]

                # Ignore very low or zero frequencies
                if LOW_FREQ < peak_freq < HIGH_FREQ:
                    # Convert frequency to note
                    note = librosa.hz_to_note(peak_freq)
                    # Store the note in the sequence
                    note_sequence.append(note)
                    print(f"Peak Frequency: {peak_freq:.2f} Hz, Note: {note}")
                else:
                    print("Frequency out of musical range, ignoring...")
            else:
                print("Low energy signal, ignoring...")

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
