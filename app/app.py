import streamlit as st
import os
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

# Add this near the top (after imports)
BASE_DIR = Path(__file__).resolve().parent.parent  # Gets project root
DATA_DIR = BASE_DIR / "data"

# Page config
st.set_page_config(layout="wide", page_title="LipBuddy", page_icon="🧠")

# Custom CSS
st.markdown(
    """
    <style>
        :root {
            --primary: #4A90E2;
            --secondary: #6E48AA;
            --accent: #9D50BB;
            --light: #F5F7FA;
            --dark: #1A1A2E;
        }
        .block-container {
            padding: 2rem 4rem;
        }
        h1, h2, h3 {
            text-align: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .rounded-box {
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(0,0,0,0.08);
            margin-bottom: 2rem;
            border: 1px solid rgba(255,255,255,0.2);
            background: white;
        }
        .prediction-box {
            padding: 25px 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #e1e9f0 100%);
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.05);
            text-align: center;
            border: 1px solid rgba(255,255,255,0.3);
            margin-top: 1rem;
        }
        .section-title {
            color: var(--primary);
            margin: 1.5rem 0;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        [data-testid="stSidebar"] {
            display: none;
        }

        .video-container {
            display: flex;
            justify-content: center;
            align-items: center;
            border-radius: 20px;
            overflow: hidden;
        }
        .gif-container {
            display: flex;
            justify-content: center;
            align-items: center;
            background: #f7f9fb;
            padding: 20px;
            border-radius: 20px;
            height: 100%;
        }
        .token-output {
            padding: 1rem;
            background: #ffffff;
            border-radius: 12px;
            border: 1px solid #ddd;
            margin-top: 1rem;
        }
        .token-output pre {
            font-size: 0.9rem;
            color: #444;
        }
        @media (max-width: 768px) {
            .block-container {
                padding: 1rem;
            }
        }
    </style>
""",
    unsafe_allow_html=True,
)

# App title
st.markdown(
    """
    <h2>👁️‍🗨️ Visible Voice 👄
</h2>
    <h3><i>“See what’s being said.”</i></h3>
""",
    unsafe_allow_html=True,
)

# Video selection
with st.container():
    st.markdown(
        """
        <div style="
            display: flex;
            justify-content: center;
            padding-left: 40px;
            padding-right: 40px;
        ">
        """,
        unsafe_allow_html=True,
    )

    video_folder = DATA_DIR / "s1"
    options = [f.name for f in video_folder.glob("*.mpg")]  # List only .mpg files
    selected_video = st.selectbox(
        "🎥 Select a video sample to analyze",
        options,
        help="Choose a video file for lip reading analysis",
    )

    st.markdown("</div>", unsafe_allow_html=True)


# Then update the file path usage:
if selected_video:
    file_path = str(video_folder / selected_video)  # Convert to string for compatibility

    # Load data and process prediction once
    with st.spinner("Loading video and running model..."):
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave("animation.gif", video, fps=10)

        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        converted_prediction = (
            tf.strings.reduce_join(num_to_char(decoder)).numpy().decode("utf-8")
        )

        os.system(f"ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y")
        video_file = open("test_video.mp4", "rb")
        video_bytes = video_file.read()

    # Layout: 3 columns
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 🎞️ Original Video")
        st.markdown(
            "<div class='rounded-box video-container', style='max_width: 100px;'>",
            unsafe_allow_html=True,
        )
        st.video(video_bytes)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("### 🔠 Token Output")
        st.write("")

        st.markdown(
            "<div class='token-output'><pre>" + str(decoder.tolist()) + "</pre></div>",
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("### 👁️ Model Prediction")
        st.markdown(
            f"""
            <div class="prediction-box" style="
                background-color: #d4edda;  /* light green */
                padding: 1rem;
                border-radius: 8px;
            ">
                <h2 style="
                    color: black;  /* dark green */
                    font-weight: 700;
                    margin: 0;
                    font-size: 3rem;
                ">
                    "{converted_prediction}"
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: var(--dark); opacity: 0.7; margin-top: 2rem;">
            <p><b>Made with ❤️ by Dhanraj Sharma</b></p>
        </div>
    """,
        unsafe_allow_html=True,
    )
