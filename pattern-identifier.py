import pyaudio
import numpy as np
import librosa
import scipy.signal as signal
import time

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

# To store the sequence of notes and their timestamps
note_sequence = []
timestamps = []

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data)
    return y

def find_fundamental_frequency(fft_result, freqs):
    # Focus on the lowest frequency with the highest amplitude
    peak_index = np.argmax(fft_result)
    peak_freq = freqs[peak_index]
    
    # Try to avoid picking harmonics by checking surrounding frequencies
    for i in range(1, 5):
        harmonic_index = peak_index // i
        if harmonic_index >= 0 and fft_result[harmonic_index] > fft_result[peak_index] * 0.8:
            peak_freq = freqs[harmonic_index]
            break
    return peak_freq

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

            # Find the fundamental frequency only if the energy is above the threshold
            if energy > ENERGY_THRESHOLD:
                peak_freq = find_fundamental_frequency(fft_result, freqs)

                # Ignore very low or zero frequencies
                if LOW_FREQ < peak_freq < HIGH_FREQ:
                    # Convert frequency to note
                    note = librosa.hz_to_note(peak_freq)

                    # Get the current timestamp
                    current_time = time.time()

                    # Store the note and timestamp in the sequence
                    note_sequence.append(note)
                    timestamps.append(current_time)

                    print(f"Timestamp: {current_time:.2f}s, Peak Frequency: {peak_freq:.2f} Hz, Note: {note}")
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

    # Process the sequence of notes to remove repetitions based on tempo
    if note_sequence:
        # Calculate time intervals between successive notes
        time_intervals = np.diff(timestamps)

        # Filter out very short intervals that may be due to noise
        filtered_intervals = [interval for interval in time_intervals if interval > 0.1]  # Adjust this threshold as needed

        # Estimate the tempo (in beats per minute)
        average_interval = np.mean(filtered_intervals) if len(filtered_intervals) > 0 else 0
        tempo = 60 / average_interval if average_interval > 0 else 0

        print(f"\nEstimated Tempo: {tempo:.2f} BPM")

        # Group notes based on time intervals and tempo
        grouped_notes = []
        previous_note = note_sequence[0]
        for i in range(1, len(note_sequence)):
            if time_intervals[i-1] > average_interval * 0.75:  # Adjust this multiplier as needed
                grouped_notes.append(previous_note)
            previous_note = note_sequence[i]

        grouped_notes.append(previous_note)  # Add the last note

        # Display the grouped sequence of notes
        print("\nGrouped Sequence of Notes Played:")
        print(" -> ".join(grouped_notes))
    else:
        print("\nNo valid notes were detected.")
