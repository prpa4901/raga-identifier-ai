import streamlit as st
import pyaudio
import wave
import os
import time
import requests  # Make sure requests is imported for API calls

# Initialize the Streamlit app
st.title("AI Chat Interface-Raga Identifier")

# Chat Interface
st.write("Chat with the AI about anything, including raga identification.")
user_input = st.text_input("You: ", "Ask me anything...")

# Recording state
if "recording" not in st.session_state:
    st.session_state.recording = False
if "frames" not in st.session_state:
    st.session_state.frames = []
if "stream" not in st.session_state:
    st.session_state.stream = None
if "audio_interface" not in st.session_state:
    st.session_state.audio_interface = None

# Setup pyaudio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
AUDIO_FILE = "shared-data/output.wav"
RECORD_SECONDS = 15  # Set recording duration to 10 seconds

def start_recording():
    st.session_state.audio_interface = pyaudio.PyAudio()
    st.session_state.stream = st.session_state.audio_interface.open(format=FORMAT,
                                                                    channels=CHANNELS,
                                                                    rate=RATE,
                                                                    input=True,
                                                                    frames_per_buffer=CHUNK)
    st.session_state.frames = []
    st.session_state.recording = True
    st.write("Listening...")

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = st.session_state.stream.read(CHUNK)
        st.session_state.frames.append(data)

    stop_recording()

def stop_recording():
    stream = st.session_state.stream
    frames = st.session_state.frames
    
    stream.stop_stream()
    stream.close()
    st.session_state.audio_interface.terminate()
    
    with wave.open(AUDIO_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(st.session_state.audio_interface.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    st.session_state.recording = False
    st.session_state.stream = None
    st.session_state.audio_interface = None
    st.write("Recording completed and saved as output.wav")

    # Provide a playback option
    st.audio(AUDIO_FILE, format='audio/wav')

# Start Recording Button
if st.button("Start Recording") and not st.session_state.recording:
    start_recording()

# Send to AI Button
if os.path.exists(AUDIO_FILE) and st.button("Send to AI"):
    if user_input:
        # Example: Call an AI service for a general chat
        curr_path = os.getcwd()
        file_path = os.path.join(os.getcwd(), AUDIO_FILE)
        response = requests.post(
            "http://localhost:6000/api/identify-raga",
            json={"prompt": user_input, "file":True}
        )
        print(response)
        time.sleep(2)
        raga_name = response.json().get("raga", "Unknown Raga")
        # raga_name['response']
        st.write(f"AI: {raga_name}")
    else:
        st.write("Please provide a query before sending to AI.")
