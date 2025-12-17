import streamlit as st
from openai import OpenAI
import json
from datetime import datetime

st.set_page_config(page_title="English Coaching Panel", layout="centered")
st.title("🎧 English Coaching Panel")

# ---------- Session State ----------
if "lesson_active" not in st.session_state:
    st.session_state.lesson_active = False

if "day_no" not in st.session_state:
    st.session_state.day_no = 1

if "lesson_no" not in st.session_state:
    st.session_state.lesson_no = 1

# ---------- Settings ----------
with st.expander("⚙️ Settings"):
    speaking_minutes = st.slider("Speaking target (minutes)", 5, 20, 10)
    min_sentences = st.slider("Min sentences (fallback)", 5, 20, 8)
    speed = st.selectbox("Speaking speed", ["Slow", "Medium", "Natural"])
    report_length = st.selectbox("Report length", ["Short", "Standard", "Long"])

st.divider()

# ---------- Credentials ----------
api_key = st.text_input("🔑 OpenAI API Key", type="password")
system_prompt = st.text_area("🧠 SYSTEM PROMPT", height=240, placeholder="Paste V7 prompt here")

client = OpenAI(api_key=api_key) if api_key else None

# ---------- Lesson Start / Stop ----------
st.subheader("▶️ Lesson Start")

colA, colB, colC = st.columns([1, 1, 2])

start_disabled = (not api_key) or (not system_prompt) or st.session_state.lesson_active
end_disabled = not st.session_state.lesson_active

if colA.button("START LESSON", disabled=start_disabled):
    st.session_state.lesson_active = True
    st.success("Lesson is ACTIVE ✅")
    st.info("Use TALK / STOP to speak. The assistant will respond after STOP.")

if colB.button("END LESSON", disabled=end_disabled):
    st.session_state.lesson_active = False
    st.session_state.lesson_no += 1
    st.success("Lesson ended ✅ (Next lesson number prepared)")

colC.markdown(
    f"**Status:** {'ACTIVE' if st.session_state.lesson_active else 'INACTIVE'}  \n"
    f"**Day:** {st.session_state.day_no}  \n"
    f"**Lesson No:** {st.session_state.lesson_no}"
)

st.divider()

# ---------- Talk / Stop controls ----------
st.subheader("🎙️ Talk (Manual)")

col1, col2 = st.columns(2)
talk = col1.button("TALK", disabled=not st.session_state.lesson_active)
stop = col2.button("STOP", disabled=not st.session_state.lesson_active)

if talk:
    st.info("Listening... Speak now. When you finish, press STOP.")

# ---------- Assistant response ----------
if stop and client and system_prompt and st.session_state.lesson_active:
    # Minimal instruction to let your V7 system control the flow
    user_msg = (
        "Continue the lesson according to the SYSTEM PROMPT rules. "
        "If the learner spoke briefly, ask follow-up questions. "
        "Speak simply and slowly if needed. "
        "Do NOT end early. "
        "At the end of the lesson, produce written feedback and the required JSON."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        st.subheader("🔊 Assistant")
        st.write(response.choices[0].message.content)

    except Exception as e:
        st.error(f"OpenAI error: {e}")

st.divider()

# ---------- Report area (placeholder; V7 will output real ones when lesson ends) ----------
st.subheader("📊 Lesson Report (Panel)")
st.caption("When the lesson completes, you will see feedback + JSON in the assistant output. "
           "You can also export a simple technical JSON below (optional).")

tech_report = {
    "date": datetime.utcnow().isoformat(),
    "day_no": st.session_state.day_no,
    "lesson_no": st.session_state.lesson_no,
    "speaking_target_minutes": speaking_minutes,
    "min_sentences": min_sentences,
    "speed": speed,
    "report_length": report_length,
    "panel_note": "This is a technical panel state snapshot, not the main V7 memory JSON."
}

st.code(json.dumps(tech_report, indent=2))
st.download_button("Download panel JSON", data=json.dumps(tech_report, indent=2), file_name="panel_state.json")
