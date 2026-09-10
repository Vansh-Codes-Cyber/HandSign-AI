# HandSign AI — Streamlit Edition

This is the original **Sign Language Recognition System** rebuilt with
[Streamlit](https://streamlit.io) instead of Flask + HTML/CSS/JS.

## What changed

| Original (Flask) | Streamlit version |
|---|---|
| `app.py` (Flask routes) | `app.py` (single Streamlit script, page picker in sidebar) |
| `templates/index.html`, `live.html`, `about.html` | Rendered directly in `app.py` with `st.*` components |
| `static/css/style.css`, `static/js/main.js` | A small `st.markdown` CSS block + native Streamlit widgets (no JS needed) |
| `camera.py` (MJPEG generator for `<img>` tag) | `camera.py` (`process_frame()` called each rerun, image shown with `st.image`) |
| `/prediction` JSON endpoint + `setInterval` polling in JS | Handled automatically by Streamlit's rerun loop |

The machine-learning pipeline itself — `models/hand_sign_model.joblib`,
`scripts/collect_data.py`, `scripts/create_dataset.py`,
`scripts/train_model.py`, `scripts/predict.py` — is **unchanged**, since
none of that is UI code.

## Project structure

```
HandSign_AI_Streamlit/
├── app.py                 # Streamlit app (Home / Live Demo / About)
├── camera.py               # Model loading + per-frame hand detection & prediction
├── requirements.txt
├── models/
│   └── hand_sign_model.joblib
├── output/
│   └── recognized_signs.txt   # log of recognized signs (written at runtime)
└── scripts/                # unchanged data collection / training scripts
    ├── collect_data.py
    ├── create_dataset.py
    ├── train_model.py
    └── predict.py
```

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens the app in your browser. Use the sidebar to switch between:

- **Home** — landing page with feature highlights
- **Live Demo** — opens your webcam (via OpenCV), runs MediaPipe hand
  tracking + the XGBoost model, and shows the live prediction, confidence
  bar, and detection status
- **About** — project overview, tech stack, and future scope

## Notes on the Live Demo page

Streamlit has no built-in equivalent of Flask's MJPEG `Response` stream,
so the webcam loop works by:

1. Wrapping the camera/prediction logic in an `st.fragment(run_every=0.08)`
   — this re-runs just that fragment roughly every 80ms, instead of
   rerunning the whole page.
2. Each fragment run reads one frame from `cv2.VideoCapture`, runs
   MediaPipe + the model on it, and draws the landmarks/label.
3. Displaying it with `st.image`, updating the prediction/confidence/status
   placeholders in place.
4. `run_every` is only active while `st.session_state.run_camera` is
   `True`; clicking **Stop Camera** flips that flag and the fragment
   naturally stops rerunning itself.

The model, MediaPipe detector, and camera handle are each loaded once via
`st.cache_resource` so they aren't reloaded on every single frame.
