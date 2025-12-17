import streamlit as st
from openai import OpenAI
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import soundfile as sf
from io import BytesIO
import streamlit.components.v1 as components
import time
from collections import Counter

# =========================
# Page Config
# =========================
st.set_page_config(page_title="English Coaching Panel", layout="centered")
st.title("🎧 English Coaching Panel")

# =========================
# Session State Initialization
# =========================
if "lesson_active" not in st.session_state:
    st.session_state.lesson_active = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audio_chunks" not in st.session_state:
    st.session_state.audio_chunks = []
if "prev_playing" not in st.session_state:
    st.session_state.prev_playing = False
if "processing_stop" not in st.session_state:
    st.session_state.processing_stop = False
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""
if "last_assistant" not in st.session_state:
    st.session_state.last_assistant = ""

# =========================
# Sidebar Settings
# =========================
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    system_prompt = st.text_area("🧠 System Prompt", height=300)
    
    st.divider()
    speaking_minutes = st.slider("Target Minutes", 5, 20, 10)
    auto_speak = st.toggle("🔊 Auto Speak", value=True)
    min_audio_seconds = st.slider("Min Audio Guard", 0.1, 2.0, 0.5)
    stop_drain_ms = st.slider("Drain Window (ms)", 500, 3000, 1500)

client = OpenAI(api_key=api_key) if api_key else None

# =========================
# Helper Functions
# =========================
def speak_in_browser(text: str):
    if not text: return
    # JavaScript içindeki tırnak hatalarını önlemek için temizlik
    safe_text = text.replace("`", "'").replace("\n", " ").replace('"', "'")
    components.html(f"""
        <script>
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                const u = new SpeechSynthesisUtterance(`{safe_text}`);
                u.lang = 'en-US';
                u.rate = 0.9;
                window.speechSynthesis.speak(u);
            }}
        </script>
    """, height=0)

def assistant_call(user_text: str):
    if not client: return "Error: No API Key"
    st.session_state.messages.append({"role": "user", "content": user_text})
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.messages,
        )
        out = resp.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": out})
        st.session_state.last_assistant = out
        return out
    except Exception as e:
        return f"API Error: {str(e)}"

# =========================
# Main UI Logic
# =========================
if not st.session_state.lesson_active:
    st.info("Welcome! Please paste your System Prompt and API Key in the sidebar, then press START.")
    if st.button("🚀 START LESSON", use_container_width=True):
        if api_key and system_prompt:
            st.session_state.lesson_active = True
            st.session_state.messages = [{"role": "system", "content": system_prompt}]
            # Başlangıç mesajını al
            initial_msg = assistant_call("START LESSON pressed. Begin the lesson according to rules.")
            st.rerun()
        else:
            st.error("Missing API Key or System Prompt!")

else:
    # --- Lesson is ACTIVE ---
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("⏹️ END"):
            st.session_state.lesson_active = False
            st.rerun()

    # Display Conversation
    if st.session_state.last_assistant:
        with st.chat_message("assistant"):
            st.write(st.session_state.last_assistant)
            # Sayfa yüklendiğinde asistan otomatik konuşsun
            if auto_speak and not st.session_state.prev_playing and not st.session_state.last_transcript:
                speak_in_browser(st.session_state.last_assistant)

    st.divider()

    # Audio Recorder
    ctx = webrtc_streamer(
        key="english-coach-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=4096,
        media_stream_constraints={"video": False, "audio": True},
    )

    is_playing = bool(ctx and ctx.state.playing)

    # Record frames while the mic is active
    if is_playing and ctx.audio_receiver:
        try:
            frames = ctx.audio_receiver.get_frames(timeout=0.1)
            for f in frames:
                # Convert to mono float32
                arr = f.to_ndarray().flatten().astype(np.float32)
                # Int16 -> Float32 conversion if necessary
                if f.format.sample_format == 's16':
                    arr = arr / 32768.0
                st.session_state.audio_chunks.append((arr, f.sample_rate))
        except:
            pass

    # Handle the STOP event
    if st.session_state.prev_playing and not is_playing:
        st.session_state.processing_stop = True
        
        # 1. Drain window (Wait for late packets)
        with st.spinner("Catching final words..."):
            time.sleep(stop_drain_ms / 1000.0)
            if ctx.audio_receiver:
                try:
                    frames = ctx.audio_receiver.get_frames(timeout=0.1)
                    for f in frames:
                        arr = f.to_ndarray().flatten().astype(np.float32)
                        if f.format.sample_format == 's16': arr = arr / 32768.0
                        st.session_state.audio_chunks.append((arr, f.sample_rate))
                except: pass

        # 2. Calculate duration
        if st.session_state.audio_chunks:
            total_samples = sum(len(x[0]) for x in st.session_state.audio_chunks)
            avg_sr = st.session_state.audio_chunks[0][1]
            duration = total_samples / avg_sr
        else:
            duration = 0

        # 3. Guard & Process
        if duration < min_audio_seconds:
            st.warning(f"Audio too short ({duration:.2f}s). Please speak clearly and wait a moment before stopping.")
            st.session_state.audio_chunks = []
            st.session_state.processing_stop = False
        else:
            with st.spinner("Thinking..."):
                # Build WAV
                all_audio = np.concatenate([x[0] for x in st.session_state.audio_chunks])
                audio_i16 = (np.clip(all_audio, -1.0, 1.0) * 32767).astype(np.int16)
                wav_io = BytesIO()
                sf.write(wav_io, audio_i16, avg_sr, format="WAV", subtype="PCM_16")
                wav_io.seek(0)
                wav_io.name = "input.wav"

                try:
                    # Transcribe
                    transcript = client.audio.transcriptions.create(model="whisper-1", file=wav_io).text
                    st.session_state.last_transcript = transcript
                    
                    # Call AI
                    ai_response = assistant_call(f"Learner says: {transcript}")
                    if auto_speak:
                        speak_in_browser(ai_response)
                        
                except Exception as e:
                    st.error(f"Error processing audio: {e}")

            # Reset for next turn
            st.session_state.audio_chunks = []
            st.session_state.processing_stop = False
            st.rerun()

    st.session_state.prev_playing = is_playing

    if st.session_state.last_transcript:
        with st.chat_message("user"):
            st.write(st.session_state.last_transcript)

# History Section (Collapsible)
if st.session_state.messages:
    with st.expander("Full Conversation History"):
        for m in st.session_state.messages:
            st.text(f"{m['role'].upper()}: {m['content']}")
