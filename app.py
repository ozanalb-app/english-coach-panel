import streamlit as st
from openai import OpenAI
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import soundfile as sf
from io import BytesIO
import streamlit.components.v1 as components
import time
from collections import Counter

st.set_page_config(page_title="English Coaching Panel", layout="centered")
st.title("🎧 English Coaching Panel")

# =========================
# Session State
# =========================
if "lesson_active" not in st.session_state:
    st.session_state.lesson_active = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audio_chunks" not in st.session_state:
    st.session_state.audio_chunks = []
if "audio_sr" not in st.session_state:
    st.session_state.audio_sr = 48000
if "prev_playing" not in st.session_state:
    st.session_state.prev_playing = False
if "processing_stop" not in st.session_state:
    st.session_state.processing_stop = False
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""
if "last_assistant" not in st.session_state:
    st.session_state.last_assistant = ""
if "last_audio_debug" not in st.session_state:
    st.session_state.last_audio_debug = {}

# =========================
# Settings
# =========================
with st.expander("⚙️ Settings"):
    speaking_minutes = st.slider("Speaking target (minutes)", 5, 20, 10)
    min_sentences = st.slider("Min sentences (fallback)", 5, 20, 8)
    speed = st.selectbox("Speaking speed", ["Slow", "Medium", "Natural"])
    report_length = st.selectbox("Report length", ["Short", "Standard", "Long"])
    auto_speak = st.toggle("🔊 Auto speak assistant (browser voice)", value=True)

    st.divider()
    show_audio_debug = st.toggle("🧪 Show audio debug", value=True) # Varsayılan True yapıldı
    min_audio_seconds = st.slider("Min audio seconds (guard)", 0.1, 3.0, 0.4, 0.1)
    stop_drain_ms = st.slider("STOP drain window (ms)", 0, 3000, 1500, 50) # Drain window artırıldı

st.divider()

# =========================
# Credentials
# =========================
api_key = st.text_input("🔑 OpenAI API Key", type="password")
system_prompt = st.text_area("🧠 SYSTEM PROMPT", height=240, placeholder="Paste V7.1 prompt here")

client = OpenAI(api_key=api_key) if api_key else None

# =========================
# Helpers
# =========================
def speak_in_browser(text: str, rate: float = 0.9):
    safe = (text or "").replace("\\", "\\\\").replace("`", "\\`").replace("</", "<\\/")
    components.html(
        f"""
        <script>
          const txt = `{safe}`;
          if (txt && "speechSynthesis" in window) {{
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(txt);
            u.rate = {rate};
            window.speechSynthesis.speak(u);
          }}
        </script>
        """,
        height=0,
    )

def assistant_call(user_text: str) -> str:
    try:
        st.session_state.messages.append({"role": "user", "content": user_text})
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.messages,
        )
        out = resp.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": out})
        st.session_state.last_assistant = out
        return out
    except Exception as e:
        return f"AI Error: {str(e)}"

def normalize_frame_to_mono_float32(frame) -> tuple[np.ndarray, dict]:
    arr = frame.to_ndarray()
    info = {"orig_dtype": str(arr.dtype)}
    if arr.ndim == 2:
        arr = arr.mean(axis=1 if arr.shape[1] < arr.shape[0] else 0)
    if arr.dtype != np.float32:
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
        else:
            arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    arr = np.clip(arr, -1.0, 1.0)
    return arr, info

def get_frame_sr(frame) -> int:
    return int(getattr(frame, "sample_rate", 48000))

def compute_total_seconds(chunks: list[tuple[np.ndarray, int]]) -> float:
    return sum(len(samples) / sr for samples, sr in chunks if sr > 0)

def choose_write_sr(chunks: list[tuple[np.ndarray, int]]) -> int:
    srs = [sr for _, sr in chunks]
    return Counter(srs).most_common(1)[0][0] if srs else 48000

def build_wav_pcm16(chunks: list[tuple[np.ndarray, int]]) -> tuple[bytes, int]:
    write_sr = choose_write_sr(chunks)
    audio = np.concatenate([s for (s, _) in chunks]).astype(np.float32)
    audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    wav_io = BytesIO()
    sf.write(wav_io, audio_i16, samplerate=write_sr, format="WAV", subtype="PCM_16")
    return wav_io.getvalue(), write_sr

def transcribe_wav_bytes(wav_bytes: bytes) -> str:
    bio = BytesIO(wav_bytes)
    bio.name = "speech.wav"
    # whisper-1 kullanımı zorunlu
    tr = client.audio.transcriptions.create(
        model="whisper-1", 
        file=bio,
        language="en",
        prompt="English transcription for a language learner.",
    )
    return tr.text

def drain_remaining_frames(ctx, max_ms: int):
    if not ctx or not ctx.audio_receiver: return 0
    drained = 0
    t_end = time.time() + (max_ms / 1000.0)
    while time.time() < t_end:
        try:
            frames = ctx.audio_receiver.get_frames(timeout=0.05)
            if frames:
                for f in frames:
                    sr = get_frame_sr(f)
                    arr, _ = normalize_frame_to_mono_float32(f)
                    st.session_state.audio_chunks.append((arr, sr))
                    drained += 1
            else:
                time.sleep(0.05)
        except: break
    return drained

# =========================
# Lesson controls
# =========================
st.subheader("▶️ Lesson")
colA, colB = st.columns(2)

if colA.button("START LESSON", disabled=st.session_state.lesson_active or not api_key):
    st.session_state.lesson_active = True
    st.session_state.messages = [{"role": "system", "content": system_prompt}]
    st.session_state.audio_chunks = []
    first = assistant_call("START LESSON pressed. Begin Level Assessment A2-B2. No praise. Max 2 sentences.")
    st.rerun()

if colB.button("END LESSON", disabled=not st.session_state.lesson_active):
    st.session_state.lesson_active = False
    st.rerun()

st.divider()

# =========================
# Speak (STT)
# =========================
if st.session_state.lesson_active:
    st.subheader("🎙️ Speak (STT)")
    
    ctx = webrtc_streamer(
        key="stt-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=4096, # Arabelleği büyüttük
        media_stream_constraints={"video": False, "audio": True},
    )

    is_playing = bool(ctx and ctx.state.playing)

    # Ses verisini anlık yakala
    if ctx and is_playing and ctx.audio_receiver:
        try:
            frames = ctx.audio_receiver.get_frames(timeout=0.1)
            for f in frames:
                sr = get_frame_sr(f)
                arr, info = normalize_frame_to_mono_float32(f)
                st.session_state.audio_chunks.append((arr, sr))
        except:
            pass

    # STOP Algılama
    just_stopped = (st.session_state.prev_playing and not is_playing)
    st.session_state.prev_playing = is_playing

    if just_stopped and not st.session_state.processing_stop:
        st.session_state.processing_stop = True
        drain_remaining_frames(ctx, stop_drain_ms)
        
        total_sec = compute_total_seconds(st.session_state.audio_chunks)
        
        if total_sec < min_audio_seconds:
            st.warning(f"Audio too short ({total_sec:.2f}s). Talk more and wait 1s before stopping.")
            st.session_state.audio_chunks = []
            st.session_state.processing_stop = False
        else:
            with st.spinner("Processing speech..."):
                try:
                    wav_b, _ = build_wav_pcm16(st.session_state.audio_chunks)
                    txt = transcribe_wav_bytes(wav_b)
                    st.session_state.last_transcript = txt
                    
                    ai_resp = assistant_call(f"User says: {txt}. Settings: Target {speaking_minutes}m. Continue lesson.")
                    if auto_speak:
                        speak_in_browser(ai_resp)
                except Exception as e:
                    st.error(f"Error: {e}")
                
                st.session_state.audio_chunks = []
                st.session_state.processing_stop = False
                st.rerun()

    if st.session_state.last_transcript:
        st.info(f"You: {st.session_state.last_transcript}")
    if st.session_state.last_assistant:
        st.write(f"**Assistant:** {st.session_state.last_assistant}")

st.divider()
st.subheader("📜 History")
for m in st.session_state.messages[1:]:
    st.text(f"{m['role'].capitalize()}: {m['content']}")
