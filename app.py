import streamlit as st
import numpy as np
import joblib
import os
import math
import keras

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Wildlife Anomaly Detector",
    page_icon="🐘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg: #0A0F0D;
    --surface: #111A15;
    --border: #1E3024;
    --accent: #3DFF7A;
    --accent2: #FFB830;
    --accent3: #FF4F6B;
    --text: #E8F5EC;
    --muted: #6B8F74;
    --card: #131F17;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.1;
    background: linear-gradient(135deg, var(--accent) 0%, #7FFFD4 50%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent);
    border-radius: 3px 0 0 3px;
}

.metric-card .label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--text);
}

.result-banner {
    border-radius: 16px;
    padding: 2rem 2.4rem;
    text-align: center;
    margin: 1.5rem 0;
    font-family: 'Syne', sans-serif;
}

.result-normal {
    background: linear-gradient(135deg, #0D2E1A 0%, #102A1E 100%);
    border: 2px solid var(--accent);
}

.result-anomaly {
    background: linear-gradient(135deg, #2E0D16 0%, #2A1010 100%);
    border: 2px solid var(--accent3);
}

.result-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
}

.result-normal .result-title { color: var(--accent); }
.result-anomaly .result-title { color: var(--accent3); }

.result-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
}

.prob-bar-wrap {
    background: var(--border);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin: 0.8rem 0;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.5s ease;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}

.step-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}

.step-num {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent);
    letter-spacing: 2px;
    text-transform: uppercase;
}

.tag {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 1px;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 4px;
    text-transform: uppercase;
}

.tag-normal { background: #0D2E1A; color: var(--accent); border: 1px solid var(--accent); }
.tag-anomaly { background: #2E0D16; color: var(--accent3); border: 1px solid var(--accent3); }

div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] select {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    border-radius: 8px !important;
}

div.stButton > button {
    background: var(--accent) !important;
    color: #0A0F0D !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    letter-spacing: 1px;
    width: 100%;
    transition: opacity 0.2s;
}

div.stButton > button:hover { opacity: 0.85 !important; }

[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.8rem 1rem !important;
}

.stAlert { border-radius: 10px !important; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load model & scaler ─────────────────────────────────────────────────────

@st.cache_resource
def load_assets():
    model_path = "best_embed_model.keras"
    scaler_path = "scaler.pkl"
    model, scaler = None, None
    if os.path.exists(model_path):
        try:
            model = keras.saving.load_model(model_path)
        except Exception as e:
            st.error(f"Model load error: {e}")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_assets()

ELEPHANT_TO_IDX = {"LA11": 0, "LA12": 1, "LA13": 2, "LA14": 3}
WINDOW = 24
FEATURES = ["location-lat", "location-long", "speed_kmh", "distance_km"]
THRESHOLD = 0.6

# ── Helpers ─────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def build_window_from_points(points, scaler):
    raw = np.array([[p["lat"], p["lon"], p["speed_kmh"], p["distance_km"]] for p in points], dtype=np.float32)
    if scaler is not None:
        raw = scaler.transform(raw)
    return raw.reshape(1, WINDOW, 4)

def predict(window, elephant_id):
    eid = np.array([ELEPHANT_TO_IDX.get(elephant_id, 0)], dtype=np.int32)
    prob = float(model.predict([window, eid], verbose=0)[0][0])
    label = int(prob > THRESHOLD)
    return prob, label

# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.6rem;letter-spacing:3px;color:#6B8F74;text-transform:uppercase;margin-bottom:0.5rem'>System Status</div>", unsafe_allow_html=True)

    if model is not None:
        st.success("✓ Model loaded")
    else:
        st.error("✗ Model not found — place best_embed_model.keras in app folder")

    if scaler is not None:
        st.success("✓ Scaler loaded")
    else:
        st.warning("⚠ scaler.pkl not found — raw values used")

    st.markdown("---")
    st.markdown("""
    <div class='section-label'>About the Model</div>
    <div style='font-family:Space Mono,monospace;font-size:0.72rem;color:#6B8F74;line-height:1.7'>
    <b style='color:#E8F5EC'>LSTM + Embedding</b><br>
    Learns a per-elephant identity embedding.<br><br>
    <b style='color:#E8F5EC'>Window:</b> 24 timesteps<br>
    <b style='color:#E8F5EC'>Features:</b> lat, lon, speed, distance<br>
    <b style='color:#E8F5EC'>Threshold:</b> 0.6
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-label'>Elephants</div>", unsafe_allow_html=True)
    for eid in ["LA11", "LA12", "LA13", "LA14"]:
        st.markdown(f"<span class='tag tag-normal'>{eid}</span>", unsafe_allow_html=True)

# ── Hero ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='padding-top:0.5rem'>
    <div class='hero-title'>🐘 Wildlife Anomaly<br>Detection System</div>
    <div class='hero-sub'>Etosha Elephant GPS · LSTM + Identity Embedding · Real-Time Inference</div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🎯 Single Window Predict", "⚡ Quick Manual Entry", "📖 How It Works"])

# ═══════════════════════════════════════════════════════════════════
# TAB 1
# ═══════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("<div class='section-label'>Elephant Identity & Scenario</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        elephant = st.selectbox("Elephant ID", ["LA11", "LA12", "LA13", "LA14"], key="t1_eid")
    with c2:
        scenario = st.selectbox(
            "Scenario preset",
            [
                "Normal Foraging (calm, slow)",
                "Moderate Roaming (moderate speed)",
                "High-Speed Burst (anomalous)",
                "Erratic Zigzag (anomalous)",
                "Custom — enter base values below",
            ],
            key="t1_scenario",
        )

    SCENARIO_SEEDS = {
        "Normal Foraging (calm, slow)": dict(base_lat=-18.9, base_lon=15.9, speed_mu=1.5, speed_sigma=0.4, dist_mu=1.8, dist_sigma=0.4),
        "Moderate Roaming (moderate speed)": dict(base_lat=-18.85, base_lon=16.1, speed_mu=3.5, speed_sigma=0.7, dist_mu=4.0, dist_sigma=0.9),
        "High-Speed Burst (anomalous)": dict(base_lat=-18.7, base_lon=15.7, speed_mu=14.0, speed_sigma=2.5, dist_mu=16.0, dist_sigma=3.0),
        "Erratic Zigzag (anomalous)": dict(base_lat=-19.0, base_lon=16.3, speed_mu=9.0, speed_sigma=4.5, dist_mu=10.0, dist_sigma=5.0),
    }

    if scenario != "Custom — enter base values below":
        seed = SCENARIO_SEEDS[scenario]
        base_lat, base_lon = seed["base_lat"], seed["base_lon"]
        speed_mu, speed_sigma = seed["speed_mu"], seed["speed_sigma"]
        dist_mu, dist_sigma = seed["dist_mu"], seed["dist_sigma"]
    else:
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1: base_lat = st.number_input("Base Latitude", value=-18.9, format="%.4f")
        with cc2: base_lon = st.number_input("Base Longitude", value=15.9, format="%.4f")
        with cc3: speed_mu = st.number_input("Avg Speed km/h", value=2.0, min_value=0.0)
        with cc4: speed_sigma = st.number_input("Speed Variation", value=0.5, min_value=0.0)
        dist_mu = speed_mu * 1.2
        dist_sigma = speed_sigma * 1.2

    if st.button("🔍 Run Detection", key="t1_run"):
        if model is None:
            st.error("Model not loaded. Check sidebar.")
        else:
            rng = np.random.default_rng(42)
            speeds = np.clip(rng.normal(speed_mu, speed_sigma, WINDOW), 0, None)
            dists = np.clip(rng.normal(dist_mu, dist_sigma, WINDOW), 0, None)
            lats = base_lat + np.cumsum(rng.normal(0, 0.002, WINDOW))
            lons = base_lon + np.cumsum(rng.normal(0, 0.002, WINDOW))
            points = [{"lat": lats[i], "lon": lons[i], "speed_kmh": speeds[i], "distance_km": dists[i]} for i in range(WINDOW)]
            window_arr = build_window_from_points(points, scaler)
            prob, label = predict(window_arr, elephant)

            if label == 0:
                st.markdown(f"""<div class='result-banner result-normal'>
                    <div class='result-title'>✓ NORMAL MOVEMENT</div>
                    <div class='result-sub'>No anomaly detected · Confidence {(1-prob)*100:.1f}% normal</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class='result-banner result-anomaly'>
                    <div class='result-title'>⚠ ANOMALY DETECTED</div>
                    <div class='result-sub'>Unusual movement pattern · Anomaly confidence {prob*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""<div class='metric-card'>
                    <div class='label'>Anomaly Probability</div>
                    <div class='value' style='color:{"#FF4F6B" if label else "#3DFF7A"}'>{prob*100:.1f}%</div>
                    <div class='prob-bar-wrap'><div class='prob-bar-fill' style='width:{prob*100:.1f}%;background:{"#FF4F6B" if label else "#3DFF7A"}'></div></div>
                </div>""", unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""<div class='metric-card'>
                    <div class='label'>Normal Probability</div>
                    <div class='value' style='color:#3DFF7A'>{(1-prob)*100:.1f}%</div>
                    <div class='prob-bar-wrap'><div class='prob-bar-fill' style='width:{(1-prob)*100:.1f}%;background:#3DFF7A'></div></div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<div class='section-label'>Window Statistics (24 timesteps)</div>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Speed (km/h)", f"{np.mean(speeds):.2f}")
            m2.metric("Max Speed (km/h)", f"{np.max(speeds):.2f}")
            m3.metric("Avg Distance (km)", f"{np.mean(dists):.2f}")
            m4.metric("Elephant ID", elephant)

            with st.expander("📊 View raw window data"):
                import pandas as pd
                df_show = pd.DataFrame({"Timestep": range(1, 25), "Lat": lats, "Lon": lons, "Speed (km/h)": speeds, "Distance (km)": dists})
                st.dataframe(df_show.style.format({"Lat": "{:.5f}", "Lon": "{:.5f}", "Speed (km/h)": "{:.2f}", "Distance (km)": "{:.2f}"}), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 2
# ═══════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.72rem;color:#6B8F74;margin-bottom:1.2rem;line-height:1.6'>Enter a representative GPS reading. The app replicates it across a 24-step window with small noise.</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Input Parameters</div>", unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        q_elephant = st.selectbox("Elephant ID", ["LA11", "LA12", "LA13", "LA14"], key="q_eid")
    with r1c2:
        q_lat = st.number_input("Latitude", value=-18.92, format="%.5f", key="q_lat")

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        q_lon = st.number_input("Longitude", value=15.87, format="%.5f", key="q_lon")
    with r2c2:
        q_speed = st.number_input("Speed (km/h)", value=2.5, min_value=0.0, key="q_speed")
    with r2c3:
        q_dist = st.number_input("Distance (km)", value=3.0, min_value=0.0, key="q_dist")

    noise_level = st.slider("Window noise level (0 = identical points)", 0.0, 1.0, 0.15, key="q_noise")

    if st.button("⚡ Quick Predict", key="q_run"):
        if model is None:
            st.error("Model not loaded.")
        else:
            rng = np.random.default_rng(7)
            speeds = np.clip(q_speed + rng.normal(0, noise_level * q_speed + 0.01, WINDOW), 0, None)
            dists = np.clip(q_dist + rng.normal(0, noise_level * q_dist + 0.01, WINDOW), 0, None)
            lats = q_lat + np.cumsum(rng.normal(0, 0.001, WINDOW))
            lons = q_lon + np.cumsum(rng.normal(0, 0.001, WINDOW))
            points = [{"lat": lats[i], "lon": lons[i], "speed_kmh": speeds[i], "distance_km": dists[i]} for i in range(WINDOW)]
            window_arr = build_window_from_points(points, scaler)
            prob, label = predict(window_arr, q_elephant)

            col_res, col_prob = st.columns([2, 1])
            with col_res:
                if label == 0:
                    st.markdown(f"""<div class='result-banner result-normal'>
                        <div class='result-title'>✓ NORMAL</div>
                        <div class='result-sub'>Elephant {q_elephant} · {(1-prob)*100:.1f}% confidence normal</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class='result-banner result-anomaly'>
                        <div class='result-title'>⚠ ANOMALY</div>
                        <div class='result-sub'>Elephant {q_elephant} · {prob*100:.1f}% anomaly confidence</div>
                    </div>""", unsafe_allow_html=True)
            with col_prob:
                st.markdown(f"""<div class='metric-card' style='margin-top:1.5rem'>
                    <div class='label'>Raw Probability</div>
                    <div class='value' style='color:{"#FF4F6B" if label else "#3DFF7A"}'>{prob:.4f}</div>
                    <div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#6B8F74;margin-top:0.5rem'>Threshold: 0.60</div>
                </div>""", unsafe_allow_html=True)

            gauge_pct = min(q_speed / 20.0, 1.0) * 100
            gauge_color = "#3DFF7A" if q_speed < 5 else ("#FFB830" if q_speed < 10 else "#FF4F6B")
            st.markdown(f"""
            <div style='margin-top:1rem;margin-bottom:0.3rem;font-family:Space Mono,monospace;font-size:0.7rem;color:#6B8F74'>
            {q_speed:.1f} km/h · Normal: 0–5 km/h · Anomalous: >10 km/h
            </div>
            <div class='prob-bar-wrap' style='height:14px'>
                <div class='prob-bar-fill' style='width:{gauge_pct:.1f}%;background:{gauge_color}'></div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 3
# ═══════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("<div class='section-label'>Pipeline Overview</div>", unsafe_allow_html=True)
    steps = [
        ("01", "Data Collection", "Movebank GPS data for 4 Etosha elephants (LA11–LA14), year 2010."),
        ("02", "Feature Engineering", "Haversine distance between consecutive GPS pings. Speed = distance / time_diff. Labels: top 5% speed → anomaly."),
        ("03", "Window Creation", "24-timestep sliding windows. Label = 1 if >20% of steps are anomalous."),
        ("04", "Scaling", "StandardScaler fit on training set, applied to all splits."),
        ("05", "Model: LSTM + Embedding", "Elephant identity embedded as 4-dim learnable vector, concatenated with movement features before LSTM."),
        ("06", "Inference", "New window → scale → [movement, id] → model → sigmoid → threshold 0.6 → label."),
    ]
    for num, title, desc in steps:
        st.markdown(f"""<div class='step-box'>
            <div class='step-num'>STEP {num}</div>
            <div style='font-family:Syne,sans-serif;font-weight:700;font-size:1rem;color:#E8F5EC;margin:0.2rem 0'>{title}</div>
            <div style='font-family:Space Mono,monospace;font-size:0.72rem;color:#6B8F74;line-height:1.6'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='section-label' style='margin-top:1.5rem'>What counts as anomalous?</div>
    <div style='font-family:Space Mono,monospace;font-size:0.72rem;color:#6B8F74;line-height:1.8'>
    Flagged when a window contains a high proportion of high-speed readings (above 95th percentile of training speed) —
    roughly above <span style='color:#FFB830'>~10 km/h</span>. Normal foraging: <span style='color:#3DFF7A'>1–5 km/h</span>.
    </div>""", unsafe_allow_html=True)
