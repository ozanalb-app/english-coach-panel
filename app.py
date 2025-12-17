import streamlit as st
from openai import OpenAI
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import soundfile as sf
from io import BytesIO
import streamlit.components.v1 as components
import time

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
    # Debug + audio guards
    show_audio_debug = st.toggle("🧪 Show audio debug", value=False)
    min_audio_seconds = st.slider("Min audio seconds (guard)", 0.2, 2.0, 0.6, 0.1)
    stop_drain_ms = st.slider("STOP drain window (ms)", 0, 1500, 700, 50)

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
    st.session_state.messages.append({"role": "user", "content": user_text})
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.messages,
    )
    out = resp.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": out})
    st.session_state.last_assistant = out
    return out

def build_wav_pcm16(chunks: list[np.ndarray], sr: int) -> bytes:
    """Build a clean PCM_16 WAV that OpenAI reliably accepts."""
    audio = np.concatenate(chunks).astype(np.float32)

    # Guard: remove NaN/inf
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip to [-1, 1] then int16
    audio = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio * 32767.0).astype(np.int16)

    wav_io = BytesIO()
    sf.write(wav_io, audio_i16, samplerate=int(sr), format="WAV", subtype="PCM_16")
    return wav_io.getvalue()

def transcribe_wav_bytes(wav_bytes: bytes) -> str:
    bio = BytesIO(wav_bytes)
    bio.name = "speech.wav"
    bio.seek(0)
    tr = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=bio,
        language="en",
        prompt="Transcribe in English. If unclear, choose closest English words.",
    )
    return getattr(tr, "text", str(tr))

def normalize_frame_to_mono_float32(frame) -> tuple[np.ndarray, dict]:
    """
    Robustly convert a WebRTC audio frame to mono float32 in [-1, 1].
    Handles ndarray shapes (channels, samples) and (samples, channels).
    """
    arr = frame.to_ndarray()
    info = {
        "orig_shape": tuple(arr.shape) if hasattr(arr, "shape") else None,
        "orig_dtype": str(arr.dtype) if hasattr(arr, "dtype") else None,
    }

    # Determine sample rate if available (handled outside too)
    # Convert to mono if needed
    if arr.ndim == 2:
        # Heuristic: if one dimension is small (1/2) and the other is large, treat small as channels
        s0, s1 = arr.shape
        if s0 in (1, 2) and s1 > 100:       # (channels, samples)
            arr = arr.mean(axis=0)
            info["mono_axis"] = "axis0"
        elif s1 in (1, 2) and s0 > 100:     # (samples, channels)
            arr = arr.mean(axis=1)
            info["mono_axis"] = "axis1"
        else:
            # Fallback: flatten safely (better than producing length-2 audio)
            arr = arr.reshape(-1)
            info["mono_axis"] = "flatten"
    elif arr.ndim > 2:
        arr = arr.reshape(-1)
        info["mono_axis"] = "flatten_ndim>2"
    else:
        info["mono_axis"] = "none"

    # Ensure float32 [-1, 1]
    if arr.dtype != np.float32:
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
            info["scale"] = "int_to_float"
        else:
            arr = arr.astype(np.float32)
            info["scale"] = "cast_float"
    else:
        info["scale"] = "already_float32"

    # If still outside range, clip
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, -1.0, 1.0)

    info["final_len"] = int(arr.shape[0]) if hasattr(arr, "shape") else None
    return arr, info

def drain_remaining_frames(ctx, max_ms: int) -> int:
    """
    After STOP, pull remaining buffered frames for a short window.
    Returns number of frames drained.
    """
    if not ctx:
        return 0
    drained = 0
    t_end = time.time() + (max_ms / 1000.0)
    while time.time() < t_end:
        try:
            frames = ctx.audio_receiver.get_frames(timeout=0.05)
        except Exception:
            frames = []
        if not frames:
            break
        for f in frames:
            # Update SR if available
            try:
                if getattr(f, "sample_rate", None):
                    st.session_state.audio_sr = int(f.sample_rate)
            except Exception:
                pass

            arr, info = normalize_frame_to_mono_float32(f)
            st.session_state.audio_chunks.append(arr)

            if show_audio_debug:
                # keep last info (lightweight)
                st.session_state.last_audio_debug = {
                    "frame_info": info,
                    "audio_sr": int(st.session_state.audio_sr or 48000),
                }

            drained += 1
    return drained

# =========================
# Lesson controls
# =========================
st.subheader("▶️ Lesson")

colA, colB = st.columns(2)
start_disabled = (not api_key) or (not system_prompt) or st.session_state.lesson_active
end_disabled = not st.session_state.lesson_active

if colA.button("START LESSON", disabled=start_disabled):
    st.session_state.lesson_active = True
    st.session_state.messages = [{"role": "system", "content": system_prompt}]
    st.session_state.audio_chunks = []
    st.session_state.prev_playing = False
    st.session_state.processing_stop = False
    st.session_state.last_transcript = ""
    st.session_state.last_assistant = ""
    st.session_state.last_audio_debug = {}

    first = assistant_call(
        "START LESSON was pressed. Begin the lesson now. "
        "VOICE style: 1–2 short sentences, then ask ONE simple question. "
        "IMPORTANT: Do NOT use praise fillers or coaching hype. "
        "Do NOT say: Great / Great job / Thank you for sharing / Well done. "
        "Be neutral and continue automatically."
    )
    st.success("Lesson ACTIVE ✅")
    st.write("**Assistant:**", first)
    if auto_speak:
        speak_in_browser(first)

if colB.button("END LESSON", disabled=end_disabled):
    st.session_state.lesson_active = False
    st.success("Lesson ended ✅")

st.divider()

# =========================
# Speak (STT) — ONE STOP ONLY
# =========================
st.subheader("🎙️ Speak (STT)")

if not st.session_state.lesson_active:
    st.info("Press START LESSON to begin.")
else:
    st.caption(
        "Press START to record, speak, then press STOP once. "
        "After STOP, it will auto-transcribe and ask the next question."
    )

    ctx = webrtc_streamer(
        key="stt-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        media_stream_constraints={"video": False, "audio": True},
    )

    if st.button("CLEAR AUDIO BUFFER"):
        st.session_state.audio_chunks = []
        st.session_state.last_transcript = ""
        st.session_state.processing_stop = False
        st.session_state.last_audio_debug = {}
        st.success("Cleared.")

    is_playing = bool(ctx and ctx.state.playing)

    # Collect frames while playing
    if ctx and is_playing:
        try:
            frames = ctx.audio_receiver.get_frames(timeout=1)
        except Exception:
            frames = []

        for f in frames:
            # Sample rate from frame (if available)
            try:
                if getattr(f, "sample_rate", None):
                    st.session_state.audio_sr = int(f.sample_rate)
            except Exception:
                pass

            arr, info = normalize_frame_to_mono_float32(f)
            st.session_state.audio_chunks.append(arr)

            if show_audio_debug:
                st.session_state.last_audio_debug = {
                    "frame_info": info,
                    "audio_sr": int(st.session_state.audio_sr or 48000),
                }

    # Detect STOP event
    just_stopped = (st.session_state.prev_playing is True) and (is_playing is False)
    st.session_state.prev_playing = is_playing

    if just_stopped and not st.session_state.processing_stop:
        st.session_state.processing_stop = True

        if not client:
            st.error("API key missing.")
            st.session_state.processing_stop = False
        elif not st.session_state.audio_chunks:
            # Try draining anyway (some browsers deliver frames right after STOP)
            drained = drain_remaining_frames(ctx, max_ms=int(stop_drain_ms))
            if not st.session_state.audio_chunks:
                st.warning("No audio captured. Please press START and speak before STOP.")
                st.session_state.processing_stop = False
            else:
                if show_audio_debug:
                    st.caption(f"Drained frames after STOP: {drained}")
        else:
            # Drain remaining buffered frames after STOP
            drained = drain_remaining_frames(ctx, max_ms=int(stop_drain_ms))
            if show_audio_debug:
                st.caption(f"Drained frames after STOP: {drained}")

        # Proceed if we have audio
        if st.session_state.processing_stop and st.session_state.audio_chunks:
            try:
                sr = int(st.session_state.audio_sr or 48000)

                # Guard: minimum duration (prevents corrupted/empty WAV)
                total_samples = int(sum(len(x) for x in st.session_state.audio_chunks))
                seconds = (total_samples / sr) if sr > 0 else 0.0

                if show_audio_debug:
                    st.session_state.last_audio_debug.update(
                        {
                            "total_samples": total_samples,
                            "seconds": float(seconds),
                            "min_required_seconds": float(min_audio_seconds),
                        }
                    )

                if seconds < float(min_audio_seconds):
                    st.warning("Audio too short. Please record a bit longer.")
                    st.session_state.audio_chunks = []
                    st.session_state.processing_stop = False
                else:
                    wav_bytes = build_wav_pcm16(st.session_state.audio_chunks, sr=sr)
                    transcript = transcribe_wav_bytes(wav_bytes)

                    st.session_state.last_transcript = transcript
                    st.success("Transcribed ✅")
                    st.write("**You (transcript):**", transcript)

                    out = assistant_call(
                        f"My spoken answer (transcribed): {transcript}\n\n"
                        f"Panel settings: speaking_target_minutes={speaking_minutes}, "
                        f"min_sentences={min_sentences}, speed={speed}, report_length={report_length}.\n"
                        "Continue the lesson according to the SYSTEM PROMPT rules. "
                        "If my answer is short, ask follow-up questions until the target is met. "
                        "IMPORTANT: Do NOT use praise fillers or coaching hype. "
                        "Be neutral and continue automatically. Ask the next question."
                    )

                    st.write("**Assistant:**", out)
                    if auto_speak:
                        speak_in_browser(out)

                    st.session_state.audio_chunks = []
                    st.session_state.processing_stop = False

            except Exception as e:
                st.error(f"STT/Assistant error: {e}")
                st.session_state.audio_chunks = []
                st.session_state.processing_stop = False

    if st.session_state.last_transcript:
        st.caption("Last transcript:")
        st.write(st.session_state.last_transcript)
    if st.session_state.last_assistant:
        st.caption("Last assistant message:")
        st.write(st.session_state.last_assistant)

    if show_audio_debug and st.session_state.last_audio_debug:
        st.subheader("🧪 Audio debug")
        st.json(st.session_state.last_audio_debug)

st.divider()

st.subheader("📜 Conversation (this lesson)")
for m in st.session_state.messages:
    if m["role"] == "assistant":
        st.markdown(f"**Assistant:** {m['content']}")
    elif m["role"] == "user":
        st.markdown(f"**You:** {m['content']}")
