import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="English Coaching Panel", layout="centered")
st.title("🎧 English Coaching Panel")

# ---------- Session State ----------
if "lesson_active" not in st.session_state:
    st.session_state.lesson_active = False

if "day_no" not in st.session_state:
    st.session_state.day_no = 1

if "lesson_no" not in st.session_state:
    st.session_state.lesson_no = 1

if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history for OpenAI

# ---------- Settings ----------
with st.expander("⚙️ Settings"):
    speaking_minutes = st.slider("Speaking target (minutes)", 5, 20, 10)
    min_sentences = st.slider("Min sentences (fallback)", 5, 20, 8)
    speed = st.selectbox("Speaking speed", ["Slow", "Medium", "Natural"])
    report_length = st.selectbox("Report length", ["Short", "Standard", "Long"])

st.divider()

# ---------- Credentials ----------
api_key = st.text_input("🔑 OpenAI API Key", type="password")
system_prompt = st.text_area("🧠 SYSTEM PROMPT", height=240, placeholder="Paste V7.1 prompt here")

client = OpenAI(api_key=api_key) if api_key else None

# ---------- Helpers ----------
def assistant_call(user_text: str):
    """Send a user turn to OpenAI and get assistant text back, keeping history."""
    st.session_state.messages.append({"role": "user", "content": user_text})
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.messages,
    )
    out = resp.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": out})
    return out

# ---------- Lesson Start / End ----------
st.subheader("▶️ Lesson Start")

colA, colB, colC = st.columns([1, 1, 2])

start_disabled = (not api_key) or (not system_prompt) or st.session_state.lesson_active
end_disabled = not st.session_state.lesson_active

if colA.button("START LESSON", disabled=start_disabled):
    st.session_state.lesson_active = True
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

    # Force the assistant to start the correct flow immediately
    first = assistant_call(
        "The lesson has started (START LESSON was pressed). "
        "Begin now with the required flow. "
        "Start with Warm-up (voice style: 1–2 short sentences) and ask your first question."
    )
    st.success("Lesson is ACTIVE ✅")
    st.subheader("🔊 Assistant")
    st.write(first)

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

# ---------- Learner input (temporary replacement for speech-to-text) ----------
st.subheader("🗣️ Your answer (type here for now)")
learner_text = st.text_area(
    "Write what you said (we will add real speech-to-text later).",
    height=120,
    disabled=not st.session_state.lesson_active
)

col1, col2 = st.columns(2)
send = col1.button("SEND ANSWER", disabled=(not st.session_state.lesson_active) or (not api_key) or (not system_prompt))
next_q = col2.button("ASK NEXT", disabled=(not st.session_state.lesson_active) or (not api_key) or (not system_prompt))

if send and learner_text.strip():
    try:
        out = assistant_call(
            f"My answer: {learner_text}\n\n"
            f"Panel settings: speaking_target_minutes={speaking_minutes}, min_sentences={min_sentences}, speed={speed}, report_length={report_length}.\n"
            "Continue the lesson according to the rules. Ask the next question."
        )
        st.subheader("🔊 Assistant")
        st.write(out)
    except Exception as e:
        st.error(f"OpenAI error: {e}")

if next_q:
    try:
        out = assistant_call(
            "Continue now. Ask the next question. Keep it simple and slow if needed."
        )
        st.subheader("🔊 Assistant")
        st.write(out)
    except Exception as e:
        st.error(f"OpenAI error: {e}")

st.divider()

# ---------- Conversation log ----------
st.subheader("📜 Conversation (this lesson)")
if st.session_state.messages:
    for m in st.session_state.messages:
        if m["role"] == "assistant":
            st.markdown(f"**Assistant:** {m['content']}")
        elif m["role"] == "user":
            st.markdown(f"**You:** {m['content']}")
