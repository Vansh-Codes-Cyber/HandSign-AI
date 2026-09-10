import streamlit as st
import camera

# ===============================
# Page Config
# ===============================

st.set_page_config(
    page_title="HandSign AI | Hand Sign Recognition System",
    page_icon="🤟",
    layout="wide",
)

# ===============================
# Light custom styling
# ===============================

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }
        .hero-badge {
            display: inline-block;
            background: #eef2ff;
            color: #4338ca;
            padding: 4px 14px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .feature-card {
            background: #ffffff;
            border: 1px solid #eee;
            border-radius: 14px;
            padding: 1.4rem;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
            height: 100%;
        }
        .feature-card h3 {
            margin: 0.5rem 0;
        }
        .prediction-box {
            text-align: center;
            font-size: 3rem;
            font-weight: 700;
            padding: 1.5rem;
            border-radius: 16px;
            background: #f5f7ff;
            border: 1px solid #e3e8ff;
        }
        .status-online {
            color: #16a34a;
            font-weight: 600;
        }
        .status-offline {
            color: #dc2626;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===============================
# Session State
# ===============================

if "run_camera" not in st.session_state:
    st.session_state.run_camera = False
if "last_saved_prediction" not in st.session_state:
    st.session_state.last_saved_prediction = None

# ===============================
# Sidebar Navigation
# ===============================

st.sidebar.markdown("## 🤟 HandSign AI")
page = st.sidebar.radio(
    "Navigate", ["Home", "Live Demo", "About"], label_visibility="collapsed"
)

# ===============================
# HOME PAGE
# ===============================

if page == "Home":
    st.markdown(
        '<span class="hero-badge">AI Powered Recognition</span>', unsafe_allow_html=True
    )
    st.title("Hand Sign Recognition, Made Simple")
    st.write(
        "A real-time hand sign recognition system built using **MediaPipe**, "
        "**OpenCV**, **XGBoost**, **Scikit-learn** and **Streamlit**."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📷 Start Recognition", use_container_width=True, type="primary"):
            st.session_state.run_camera = True
            st.info("Head to the **Live Demo** tab in the sidebar to see it in action.")
    with col2:
        st.button("Learn More → see About tab", use_container_width=True, disabled=True)

    st.divider()
    st.subheader("Key Features")
    st.caption("Everything you need for real-time hand sign recognition.")

    c1, c2, c3, c4 = st.columns(4)
    features = [
        (
            "🎥",
            "Real-Time Detection",
            "Detect and recognize hand signs instantly from a live webcam.",
        ),
        (
            "✋",
            "MediaPipe Tracking",
            "Tracks 21 hand landmarks with high accuracy and speed.",
        ),
        (
            "🧠",
            "XGBoost Model",
            "Fast machine learning predictions with excellent performance.",
        ),
        (
            "🌐",
            "Streamlit Web App",
            "Interactive browser-based interface for live recognition.",
        ),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], features):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div style="font-size:2rem">{icon}</div>
                    <h3>{title}</h3>
                    <p style="color:#555">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ===============================
# LIVE DEMO PAGE
# ===============================

elif page == "Live Demo":
    st.title("🎥 Live Recognition")

    cam_col, pred_col = st.columns([1.4, 1])

    with cam_col:
        st.subheader("Live Camera")
        frame_placeholder = st.empty()

        b1, b2 = st.columns(2)
        with b1:
            if st.button("▶ Start Camera", use_container_width=True, type="primary"):
                st.session_state.run_camera = True
        with b2:
            if st.button("⏹ Stop Camera", use_container_width=True):
                st.session_state.run_camera = False

    with pred_col:
        st.subheader("Prediction")
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()
        status_placeholder = st.empty()

        st.divider()
        st.markdown("""
            **🧠 Model:** XGBoost
            **✋ Detector:** MediaPipe
            **🌐 Framework:** Streamlit
            """)

    if st.session_state.run_camera:
        camera_image = st.camera_input(
            "Take a picture of your hand",
            key="hand_camera",
        )

        if camera_image is not None:
            try:
                frame = camera.decode_image(camera_image.getvalue())
                frame = camera.prepare_frame(frame)

                frame_rgb, prediction, confidence, status, last_saved = (
                    camera.process_frame(
                        frame,
                        st.session_state.last_saved_prediction,
                    )
                )
                st.session_state.last_saved_prediction = last_saved

                frame_placeholder.image(
                    frame_rgb,
                    channels="RGB",
                    use_container_width=True,
                )
                prediction_placeholder.markdown(
                    f'<div class="prediction-box">{prediction}</div>',
                    unsafe_allow_html=True,
                )
                confidence_placeholder.progress(
                    min(int(confidence), 100),
                    text=f"Confidence: {confidence:.1f}%",
                )

                if status == "Hand Detected":
                    status_placeholder.markdown(
                        '<span class="status-online">🟢 Hand Detected</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    status_placeholder.markdown(
                        '<span class="status-offline">🔴 No Hand Detected</span>',
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                frame_placeholder.error(f"Could not process camera image: {e}")
        else:
            frame_placeholder.info(
                "Allow camera access, then take a picture of your hand."
            )
            prediction_placeholder.markdown(
                '<div class="prediction-box">--</div>',
                unsafe_allow_html=True,
            )
            confidence_placeholder.progress(0, text="Confidence: 0%")
            status_placeholder.markdown(
                '<span class="status-offline">🔴 No Hand Detected</span>',
                unsafe_allow_html=True,
            )
    else:
        frame_placeholder.info("Camera is stopped. Click **Start Camera** to begin.")
        prediction_placeholder.markdown(
            '<div class="prediction-box">--</div>',
            unsafe_allow_html=True,
        )
        confidence_placeholder.progress(0, text="Confidence: 0%")
        status_placeholder.markdown(
            '<span class="status-offline">🔴 No Hand Detected</span>',
            unsafe_allow_html=True,
        )

# ===============================
# ABOUT PAGE
# ===============================

else:
    st.title("About This Project")
    st.caption(
        "AI-powered hand sign recognition using computer vision and machine learning."
    )

    with st.container(border=True):
        st.subheader("Project Overview")
        st.write(
            "HandSign AI is a real-time hand sign recognition system that detects hand "
            "landmarks using MediaPipe and predicts hand signs using an XGBoost machine "
            "learning model. The application is built with Streamlit and OpenCV to provide "
            "a fast and interactive web interface."
        )

    with st.container(border=True):
        st.subheader("Technologies Used")
        techs = [
            "Python",
            "Streamlit",
            "OpenCV",
            "MediaPipe",
            "XGBoost",
            "Scikit-learn",
            "Joblib",
        ]
        st.markdown("\n".join(f"- {t}" for t in techs))

    with st.container(border=True):
        st.subheader("How It Works")
        st.write(
            "Webcam → OpenCV → MediaPipe → Hand Landmarks → XGBoost Model → Prediction → Live Dashboard"
        )

    with st.container(border=True):
        st.subheader("Future Scope")
        st.markdown("""
            - Support complete sign language words
            - Add voice output
            - Improve accuracy using deep learning
            - Mobile application integration
            - Multi-hand recognition
            """)
