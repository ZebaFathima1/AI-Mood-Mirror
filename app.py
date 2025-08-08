
import streamlit as st
import cv2
from deepface import DeepFace
import os
from datetime import datetime
import pandas as pd
from PIL import Image

# ------------------ Theme ------------------
st.set_page_config(page_title="AI Mood Mirror", layout="centered")

st.markdown(
    """
    <style>
    .title {
        font-size:40px;
        font-weight:700;
        color:#ff4b4b;
        text-align:center;
        margin-bottom:10px;
    }
    .subtitle {
        text-align:center;
        color:gray;
        font-size:18px;
    }
    .emoji-box {
        font-size:60px;
        text-align:center;
        margin-top:20px;
    }
    .msg {
        font-size:20px;
        font-weight:500;
        text-align:center;
        color:#4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Header ------------------
st.markdown("<div class='title'>🪞 AI Mood Mirror</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Let AI reflect your vibe today!</div>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ Emoji Map ------------------
emoji_map = {
    'happy': ('😄', 'Cheer up! You’re glowing! 🌟'),
    'sad': ('😢', 'Hey, everything will be okay 💖'),
    'angry': ('😠', 'Breathe. You’ve got this! 💪'),
    'neutral': ('😐', 'Stay calm and centered. 🧘‍♀️'),
    'surprise': ('😮', 'Wow! What happened?! 🎉'),
    'fear': ('😨', 'Fear is temporary, power is permanent! ⚡'),
    'disgust': ('🤢', 'Shake it off! New vibe incoming 🚀'),
}

# ------------------ Webcam Button ------------------
start = st.button("📸 Capture Emotion from Webcam")

if start:
    st.info("Get ready! Capturing in 3 seconds...")
    cap = cv2.VideoCapture(0)

    for i in range(3, 0, -1):
        st.write(f"⏳ Capturing in {i}...")
        cv2.waitKey(1000)

    ret, frame = cap.read()
    img_path = "captured.jpg"
    cv2.imwrite(img_path, frame)
    cap.release()
    cv2.destroyAllWindows()

    st.success("📸 Image captured!")
    st.image(Image.open(img_path), caption="Captured Image", use_container_width=True)

    try:
        result = DeepFace.analyze(img_path=img_path, actions=["emotion"], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        emoji, message = emoji_map.get(emotion, ('✨', 'Be your best self today! 💫'))

        st.markdown(f"<div class='emoji-box'>{emoji}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='msg'>🧠 Detected Emotion: <strong>{emotion.upper()}</strong><br>{message}</div>", unsafe_allow_html=True)

        # Save to CSV silently
        log_entry = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Detected Emotion": emotion
        }

        log_path = "emotion_log.csv"
        if not os.path.exists(log_path):
            pd.DataFrame([log_entry]).to_csv(log_path, index=False)
        else:
            df = pd.read_csv(log_path)
            df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
            df.to_csv(log_path, index=False)

    except Exception as e:
        st.error(f"❌ Error: {e}"
