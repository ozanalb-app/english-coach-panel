import streamlit as st
from openai import OpenAI
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import soundfile as sf
from io import BytesIO
import streamlit.components.v1 as components

st.set_page_config(page_title="English Coaching Panel", layout="centered")
st.title("🎧 English Coaching Panel")

# ---------------- Session State ----------------
if "lesson_active" not in st.session_state:
    st.session_state.lesson_active = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audio_chunks" not in st.session_state:
    st.session_state.audio_chunks = []
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""
if "last_assistant" not in st.session_state:
    st.session_state.last_assistant = ""

# ---------------- Settings ----------------
with st.expander("⚙️ Settings"):
    speaking_minutes = st.slider("Speaking target (minutes)", 5, 20, 10)
    min_sentences = st.slider("Min sentences (fallback)", 5, 20, 8)
    speed = st.selectbox("Speaking speed", ["Slow", "Medium", "Natural"])
    report_length = st.selectbox("Report length", ["Short", "Standard", "Long"])
    auto_speak = st.toggle("🔊 Auto speak assistant (browser voice)", value=True)

st.divider()

# ---------------- Credentials ----------------
api_key = st.text_input("🔑 OpenAI API Key", type="password")
system_prompt = st.text_area("🧠 SYSTEM PROMPT", height=240, placeholder="Paste V7.1 prompt here")

client = OpenAI(api_key=api_key) if api_key else None

def speak_in_browser(text: str):
    # Free TTS in browser using SpeechSynthesis
    safe = (text or "").replace("\\", "\\\\").replace("`", "\\`").replace("</", "<\\/")
    components.html(
        f"""
        <script>
          const txt = `{safe}`;
          if (txt && "speechSynthesis" in window) {{
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(txt);
            u.rate = 0.9;  // a bit slower by default
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

def transcribe_wav_bytes(wav_bytes: bytes) -> str:
    bio = BytesIO(wav_bytes)
    bio.name = "speech.wav"  # OpenAI client expects a filename
    tr = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=bio
    )
    # SDK returns text field
    return getattr(tr, "text", str(tr))

# ---------------- Lesson Controls ----------------
st.subheader("▶️ Lesson")

colA, colB = st.columns(2)

start_disabled = (not api_key) or (not system_prompt) or st.session_state.lesson_active
end_disabled = not st.session_state.lesson_active

if colA.button("START LESSON", disabled=start_disabled):
    st.session_state.lesson_active = True
    st.session_state.messages = [{"role": "system", "content": system_prompt}]
    st.session_state.audio_chunks = []
    st.session_state.last_transcript = ""
    st.session_state.last_assistant = ""

    first = assistant_call(
        "The START LESSON button was pressed. Begin the lesson now. "
        "Start with Warm-up (voice style: 1–2 short sentences) and ask the first question."
    )
    st.success("Lesson ACTIVE ✅")
    st.write("**Assistant:**", first)
    if auto_speak:
        speak_in_browser(first)

if colB.button("END LESSON", disabled=end_disabled):
    st.session_state.lesson_active = False
    st.success("Lesson ended ✅")

st.divider()

# ---------------- Voice Capture (STT) ----------------
st.subheader("🎙️ Speak (STT)")

if not st.session_state.lesson_active:
    st.info("Press START LESSON to begin.")
else:
    st.caption("Press START LESSON once. Then speak. When you finish, press STOP & TRANSCRIBE.")

    # WebRTC audio streamer
    ctx = webrtc_streamer(
        key="stt-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        media_stream_constraints={"video": False, "audio": True},
    )

    col1, col2 = st.columns(2)
    if col1.button("CLEAR AUDIO BUFFER"):
        st.session_state.audio_chunks = []
        st.session_state.last_transcript = ""
        st.success("Cleared.")

    stop_and_tx = col2.button("STOP & TRANSCRIBE")

    # Continuously pull audio frames into buffer while streaming
    if ctx and ctx.state.playing:
        audio_frames = ctx.audio_receiver.get_frames(timeout=1)
        for f in audio_frames:
            arr = f.to_ndarray()
            # Ensure shape (channels, samples) or (samples,)
            if arr.ndim == 2:
                # convert to mono
                arr = arr.mean(axis=0)
            # Convert to float32 -1..1
            if arr.dtype != np.float32:
                # many frames come as int16
                if np.issubdtype(arr.dtype, np.integer):
                    arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
                else:
                    arr = arr.astype(np.float32)
            st.session_state.audio_chunks.append(arr)

    if stop_and_tx:
        if not client:
            st.error("API key missing.")
        elif not st.session_state.audio_chunks:
            st.warning("No audio captured yet. Speak first.")
        else:
            try:
                # Concatenate audio
                audio = np.concatenate(st.session_state.audio_chunks).astype(np.float32)
                # Write WAV to bytes
                wav_io = BytesIO()
                sf.write(wav_io, audio, samplerate=48000, format="WAV")
                wav_bytes = wav_io.getvalue()

                transcript = transcribe_wav_bytes(wav_bytes)
                st.session_state.last_transcript = transcript
                st.success("Transcribed ✅")
                st.write("**You (transcript):**", transcript)

                # Send to assistant (your V7.1 rules will drive what happens next)
                out = assistant_call(
                    f"My spoken answer (transcribed): {transcript}\n\n"
                    f"Panel settings: speaking_target_minutes={speaking_minutes}, min_sentences={min_sentences}, "
                    f"speed={speed}, report_length={report_length}.\n"
                    "Continue the lesson according to the rules. Ask the next question."
                )
                st.write("**Assistant:**", out)
                if auto_speak:
                    speak_in_browser(out)

                # Reset buffer for next turn
                st.session_state.audio_chunks = []

            except Exception as e:
                st.error(f"STT error: {e}")

st.divider()

# ---------------- Conversation log ----------------
st.subheader("📜 Conversation (this lesson)")
if st.session_state.messages:
    for m in st.session_state.messages:
        if m["role"] == "assistant":
            st.markdown(f"**Assistant:** {m['content']}")
        elif m["role"] == "user":
            st.markdown(f"**You:** {m['content']}")
