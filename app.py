import streamlit as st
from openai import OpenAI
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import soundfile as sf
from io import BytesIO
import streamlit.components.v1 as components

st.set_page_config(page_title="English Coaching Panel", layout="centered")
st.title("🎧 English Coaching Panel")

# =========================
# Session State
# =========================
if "lesson_active" not in st.session_state:
    st.session_state.lesson_active = False

if "messages" not in st.session_state:
    st.session_state.messages = []  # OpenAI chat history

if "audio_chunks" not in st.session_state:
    st.session_state.audio_chunks = []

if "prev_playing" not in st.session_state:
    st.session_state.prev_playing = False

if "processing_stop" not in st.session_state:
    st.session_state.processing_stop = False  # prevent duplicate processing on reruns

if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""

if "last_assistant" not in st.session_state:
    st.session_state.last_assistant = ""


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
    """Free TTS via browser SpeechSynthesis."""
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
    """Send user turn to OpenAI and keep chat history."""
    st.session_state.messages.append({"role": "user", "content": user_text})
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.messages,
    )
    out = resp.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": out})
    st.session_state.last_assistant = out
    return out


def transcribe_wav_bytes(wav_bytes: bytes) -> str:
    """Transcribe speech. Force English."""
    bio = BytesIO(wav_bytes)
    bio.name = "speech.wav"
    # Use language hint + prompt to bias English
    tr = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=bio,
        language="en",
        prompt="Transcribe in English. If accent is unclear, choose the closest English words."
    )
    return getattr(tr, "text", str(tr))


def build_wav_from_chunks(chunks: list[np.ndarray], sr: int = 48000) -> bytes:
    """Concat float32 chunks and write WAV bytes."""
    audio = np.concatenate(chunks).astype(np.float32)
    wav_io = BytesIO()
    sf.write(wav_io, audio, samplerate=sr, format="WAV")
    return wav_io.getvalue()


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

    # Hard instruction: no praise fillers
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
        "Use the recorder below: press **START** to record, speak, then press **STOP** once. "
        "After STOP, transcription + next assistant question will run automatically."
    )

    # Keep only the widget's own Start/Stop. NO extra transcribe button.
    ctx = webrtc_streamer(
        key="stt-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        media_stream_constraints={"video": False, "audio": True},
    )

    # Optional: clear buffer
    if st.button("CLEAR AUDIO BUFFER"):
        st.session_state.audio_chunks = []
        st.session_state.last_transcript = ""
        st.session_state.processing_stop = False
        st.success("Cleared.")

    # Collect frames while playing
    is_playing = bool(ctx and ctx.state.playing)

    if ctx and is_playing:
        # Pull frames frequently
        try:
            audio_frames = ctx.audio_receiver.get_frames(timeout=1)
        except Exception:
            audio_frames = []

        for f in audio_frames:
            arr = f.to_ndarray()
            # to mono
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            # normalize to float32 -1..1
            if arr.dtype != np.float32:
                if np.issubdtype(arr.dtype, np.integer):
                    arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
                else:
                    arr = arr.astype(np.float32)
            st.session_state.audio_chunks.append(arr)

    # Detect STOP event: prev_playing True -> now False
    just_stopped = (st.session_state.prev_playing is True) and (is_playing is False)

    # Update prev state at end of run
    st.session_state.prev_playing = is_playing

    # Auto-transcribe immediately after STOP (once)
    if just_stopped and not st.session_state.processing_stop:
        st.session_state.processing_stop = True  # lock

        if not client:
            st.error("API key missing.")
            st.session_state.processing_stop = False
        elif not st.session_state.audio_chunks:
            st.warning("No audio captured. Please press START and speak before STOP.")
            st.session_state.processing_stop = False
        else:
            try:
                wav_bytes = build_wav_from_chunks(st.session_state.audio_chunks, sr=48000)
                transcript = transcribe_wav_bytes(wav_bytes)
                st.session_state.last_transcript = transcript

                st.success("Transcribed ✅")
                st.write("**You (transcript):**", transcript)

                # Send transcript to assistant. Force neutral tone (no praise).
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

            except Exception as e:
                st.error(f"STT/Assistant error: {e}")

            # Reset buffer for next turn
            st.session_state.audio_chunks = []
            st.session_state.processing_stop = False  # unlock for next stop

    # Show last outputs
    if st.session_state.last_transcript:
        st.caption("Last transcript:")
        st.write(st.session_state.last_transcript)
    if st.session_state.last_assistant:
        st.caption("Last assistant message:")
        st.write(st.session_state.last_assistant)

st.divider()

# =========================
# Conversation log
# =========================
st.subheader("📜 Conversation (this lesson)")
for m in st.session_state.messages:
    if m["role"] == "assistant":
        st.markdown(f"**Assistant:** {m['content']}")
    elif m["role"] == "user":
        st.markdown(f"**You:** {m['content']}")
