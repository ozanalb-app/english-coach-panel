import streamlit as st
from openai import OpenAI
import json
from datetime import datetime

st.set_page_config(page_title="English Coaching Panel", layout="centered")

st.title("🎧 English Coaching Panel")

# ===== Settings =====
with st.expander("⚙️ Settings"):
    speaking_minutes = st.slider("Speaking target (minutes)", 5, 15, 10)
    min_sentences = st.slider("Min sentences (fallback)", 5, 15, 8)
    speed = st.selectbox("Speaking speed", ["Slow", "Medium", "Natural"])
    report_length = st.selectbox("Report length", ["Short", "Standard", "Long"])

st.divider()

# ===== Credentials =====
api_key = st.text_input("🔑 OpenAI API Key", type="password")
system_prompt = st.text_area("🧠 SYSTEM PROMPT", height=220, placeholder="v7.x promptu buraya yapıştır")

client = OpenAI(api_key=api_key) if api_key else None

# ===== Lesson Controls =====
st.subheader("🎙️ Lesson Control")
col1, col2 = st.columns(2)
talk = col1.button("TALK")
stop = col2.button("STOP")

if "transcript" not in st.session_state:
    st.session_state.transcript = []

if talk:
    st.info("Listening... (Speak, then press STOP when finished)")

if stop and client and system_prompt:
    # Ask assistant to continue lesson / follow-up
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Continue the lesson. Ask follow-up questions if needed to reach the speaking target."}
        ]
    )
    st.subheader("🔊 Assistant")
    st.write(response.choices[0].message.content)

    # ===== Simple Lesson Report =====
    report = {
        "date": datetime.utcnow().isoformat(),
        "speaking_target_minutes": speaking_minutes,
        "min_sentences": min_sentences,
        "speed": speed,
        "report_length": report_length,
        "coach_note": "Lesson completed. Review speaking flow and follow-ups."
    }

    st.subheader("📊 Lesson Report")
    st.write("**Coach feedback:** Focus on extending answers with reasons and examples.")

    st.subheader("📋 JSON Memory")
    st.code(json.dumps(report, indent=2))
    st.download_button("Download JSON", data=json.dumps(report, indent=2), file_name="lesson_report.json")
