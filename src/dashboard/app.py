"""
app.py
======
Streamlit Dashboard for EEG Dream Detection System

Features:
  - Upload your own EEG data OR run with synthetic demo data
  - Real-time epoch-by-epoch simulation
  - Sleep stage timeline (hypnogram)
  - REM period highlighting
  - Dream content predictions with confidence bars
  - Per-class probability breakdown

Run:
  streamlit run src/dashboard/app.py
"""

import sys
import os
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import yaml
import torch

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EEG Dream Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark neuroscience theme */
    .stApp { background-color: #0a0e1a; color: #e0e8ff; }

    .metric-card {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1428 100%);
        border: 1px solid #2a3a6a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value { font-size: 2.5rem; font-weight: bold; color: #7eb8ff; }
    .metric-label { font-size: 0.85rem; color: #8899bb; text-transform: uppercase; letter-spacing: 1px; }

    .rem-badge {
        background: linear-gradient(135deg, #4a1a7a, #7a2ab0);
        border: 1px solid #9a50d0;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        color: #e0b0ff;
        font-weight: bold;
    }
    .wake-badge { background: #2a1a0a; border: 1px solid #7a5a30; border-radius: 20px;
                  padding: 4px 14px; font-size: 0.8rem; color: #ffcc80; }

    .dream-card {
        background: linear-gradient(135deg, #1a0a2e 0%, #0a0618 100%);
        border: 1px solid #6a3a9a;
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
    }
    .dream-title { font-size: 1.1rem; font-weight: bold; color: #c08aff; }
    .confidence-text { font-size: 0.9rem; color: #9988bb; }

    div[data-testid="stSidebar"] { background-color: #0d1220; }
    .stButton>button {
        background: linear-gradient(135deg, #3a1a6a, #5a2a9a);
        color: white; border: 1px solid #7a4abb; border-radius: 8px;
        padding: 10px 24px; font-weight: bold;
    }
    h1 { color: #7eb8ff !important; }
    h2, h3 { color: #a0c8ff !important; }
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────
STAGE_NAMES  = ["Wake", "N1", "N2", "N3", "REM"]
STAGE_COLORS = {
    "Wake": "#ff9f43",
    "N1":   "#54a0ff",
    "N2":   "#5f27cd",
    "N3":   "#341f97",
    "REM":  "#ee5a24",
}
DREAM_CATEGORIES = ["Face", "Object", "Animal", "Scene", "Text", "Movement"]
DREAM_ICONS      = {"Face": "👤", "Object": "📦", "Animal": "🐾",
                    "Scene": "🏞️", "Text": "📝", "Movement": "🌊"}


# ─── Load Config ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


# ─── Synthetic Data Generator ─────────────────────────────────────────────────
def generate_demo_data(duration_min=20, fs=100, n_channels=2):
    """Generate a synthetic sleep session for the demo."""
    from src.preprocessing.generate_synthetic import make_eeg_signal

    epoch_dur = 30
    epoch_smp = epoch_dur * fs
    n_epochs  = (duration_min * 60) // epoch_dur

    # Realistic sleep hypnogram
    hypnogram = (
        [0]*3 + [1]*2 + [2]*6 + [3]*8 +   # first cycle: descent
        [2]*3 + [4]*6 +                     # first REM
        [2]*4 + [3]*6 + [2]*3 + [4]*8 +    # second REM (longer)
        [1]*2 + [4]*10 + [0]*2              # third REM + wake
    )[:n_epochs]

    chunks = []
    for stage in hypnogram:
        chunks.append(make_eeg_signal(epoch_dur, fs, stage, n_ch=n_channels))

    eeg = np.concatenate(chunks, axis=-1).astype(np.float32)
    return eeg, np.array(hypnogram), epoch_dur


# ─── Plotly Charts ────────────────────────────────────────────────────────────
def plot_hypnogram(sleep_timeline, epoch_dur=30):
    """Create an interactive hypnogram (sleep stage over time)."""
    n = len(sleep_timeline)
    times_min = [i * epoch_dur / 60 for i in range(n)]

    # Stage ordering: Wake=top, REM=bottom (clinical convention)
    display_order = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
    y_vals = [display_order[s] for s in sleep_timeline]
    y_labels = ["REM", "N3", "N2", "N1", "Wake"]

    fig = go.Figure()

    # Background shading for REM periods
    in_rem = False
    rem_start = 0
    for i, stage in enumerate(sleep_timeline):
        if stage == 4 and not in_rem:
            rem_start = times_min[i]
            in_rem = True
        elif stage != 4 and in_rem:
            fig.add_vrect(
                x0=rem_start, x1=times_min[i],
                fillcolor="rgba(238,90,36,0.15)",
                layer="below", line_width=0,
            )
            in_rem = False
    if in_rem:
        fig.add_vrect(
            x0=rem_start, x1=times_min[-1],
            fillcolor="rgba(238,90,36,0.15)",
            layer="below", line_width=0,
        )

    # Hypnogram line
    fig.add_trace(go.Scatter(
        x=times_min, y=y_vals,
        mode="lines",
        line=dict(color="#7eb8ff", width=2, shape="hv"),
        fill="tozeroy",
        fillcolor="rgba(126,184,255,0.08)",
        name="Sleep stage",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,0.8)",
        font=dict(color="#8899bb"),
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Time (minutes)", gridcolor="#1a2040", color="#8899bb"),
        yaxis=dict(
            tickvals=list(range(5)), ticktext=y_labels,
            gridcolor="#1a2040", color="#8899bb",
        ),
        showlegend=False,
    )
    return fig


def plot_eeg_snippet(epoch, fs=100, title="EEG Signal"):
    """Plot a short EEG segment."""
    n_ch, n_t = epoch.shape
    t = np.linspace(0, n_t / fs, n_t)

    fig = go.Figure()
    colors = ["#7eb8ff", "#54d4b0", "#ff9f43", "#c084fc"]

    for ch in range(n_ch):
        offset = ch * 150    # vertical spacing in µV
        fig.add_trace(go.Scatter(
            x=t, y=epoch[ch] + offset,
            mode="lines",
            line=dict(color=colors[ch % len(colors)], width=1),
            name=f"CH{ch+1}",
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,0.8)",
        font=dict(color="#8899bb"),
        height=160,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text=title, font=dict(color="#7eb8ff", size=13)),
        xaxis=dict(title="Time (s)", gridcolor="#1a2040", color="#8899bb"),
        yaxis=dict(gridcolor="#1a2040", showticklabels=False),
        showlegend=(n_ch > 1),
    )
    return fig


def plot_dream_probs(probs, category_names=None):
    """Horizontal bar chart of dream category probabilities."""
    names = category_names or DREAM_CATEGORIES[:len(probs)]
    icons = [DREAM_ICONS.get(n, "🔵") for n in names]
    labels = [f"{ic} {n}" for ic, n in zip(icons, names)]

    colors = [
        f"rgba(126,184,255,{0.3 + 0.7 * p})" for p in probs
    ]

    fig = go.Figure(go.Bar(
        x=probs, y=labels,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="#2a3a6a", width=1),
        ),
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
        textfont=dict(color="#a0c8ff"),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,0.8)",
        font=dict(color="#8899bb"),
        height=240,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(range=[0, 1.1], gridcolor="#1a2040", showticklabels=False),
        yaxis=dict(gridcolor="#1a2040", color="#c0d8ff"),
        showlegend=False,
    )
    return fig


def plot_stage_distribution(sleep_timeline):
    """Pie chart of sleep stage distribution."""
    from collections import Counter
    counts = Counter(sleep_timeline)
    stages = [STAGE_NAMES[i] for i in range(5)]
    values = [counts.get(i, 0) for i in range(5)]
    colors = [STAGE_COLORS[s] for s in stages]

    fig = go.Figure(go.Pie(
        labels=stages, values=values,
        marker=dict(colors=colors, line=dict(color="#0a0e1a", width=2)),
        hole=0.5,
        textfont=dict(color="white"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8899bb"),
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(font=dict(color="#a0c8ff")),
    )
    return fig


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    config = get_config()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <h1 style="font-size:2.2rem; margin:0;">🧠 EEG Dream Detection System</h1>
        <p style="color:#6688aa; margin:4px 0 0 0; font-size:0.95rem;">
            Sleep staging · REM detection · Dream content classification
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        data_source = st.radio(
            "Data Source",
            ["🎮 Synthetic Demo", "📂 Upload EEG (.npy)"],
            index=0,
        )

        st.markdown("---")
        st.markdown("### Pipeline Settings")

        simulate_realtime = st.toggle("Simulate real-time processing", value=True)
        sim_speed = st.slider("Simulation speed", 1, 20, 5, help="Epochs per second")

        st.markdown("---")
        st.markdown("### Model Info")
        st.markdown("""
        <small>
        **Sleep Model:** DenseSleepNet<br>
        5-class: Wake / N1 / N2 / N3 / REM<br><br>
        **Dream Model:** EEGNet<br>
        6-class: Face / Object / Animal / Scene / Text / Movement
        </small>
        """, unsafe_allow_html=True)

    # ── Data Loading ──────────────────────────────────────────────────────────
    eeg_signal   = None
    true_hypno   = None
    epoch_dur    = config["preprocessing"]["epoch_duration"]
    fs           = config["preprocessing"]["sample_rate"]
    n_ch         = config["sleep_model"]["n_channels"]

    if data_source == "🎮 Synthetic Demo":
        if st.button("▶  Generate & Run Demo", use_container_width=True):
            with st.spinner("Generating synthetic EEG session..."):
                eeg_signal, true_hypno, epoch_dur = generate_demo_data(
                    duration_min=20, fs=fs, n_channels=n_ch
                )
            st.session_state["eeg_signal"]  = eeg_signal
            st.session_state["true_hypno"]  = true_hypno
            st.session_state["epoch_dur"]   = epoch_dur
            st.session_state["processed"]   = False

    else:
        uploaded = st.file_uploader("Upload EEG .npy file", type=["npy"])
        if uploaded:
            data = np.load(uploaded)
            if data.ndim == 1:
                data = data[np.newaxis, :]
            st.session_state["eeg_signal"] = data.astype(np.float32)
            st.session_state["true_hypno"] = None
            st.session_state["epoch_dur"]  = epoch_dur
            st.session_state["processed"]  = False
            st.success(f"✅ Loaded EEG: shape {data.shape}")

    # ── Run Pipeline ──────────────────────────────────────────────────────────
    if "eeg_signal" in st.session_state and not st.session_state.get("processed", True):
        eeg_signal = st.session_state["eeg_signal"]
        epoch_dur  = st.session_state["epoch_dur"]

        epoch_smp = int(epoch_dur * fs)
        n_epochs  = eeg_signal.shape[-1] // epoch_smp

        # Try to load the actual pipeline; fall back to mock if models not trained
        try:
            from src.inference.pipeline import EEGDreamPipeline
            pipeline_obj = EEGDreamPipeline()
            use_real_model = True
        except Exception:
            use_real_model = False

        st.markdown("## 📡 Processing EEG Signal")

        # Progress + live display containers
        progress_bar    = st.progress(0)
        status_text     = st.empty()
        live_col1, live_col2 = st.columns([2, 1])
        with live_col1:
            eeg_chart_slot  = st.empty()
        with live_col2:
            stage_badge_slot = st.empty()

        sleep_timeline    = []
        dream_predictions = []
        all_sleep_probs   = []

        for i in range(n_epochs):
            epoch = eeg_signal[:, i * epoch_smp: (i + 1) * epoch_smp]

            if use_real_model:
                result = pipeline_obj.process_epoch(epoch, i)
                stage       = result.sleep_stage
                stage_name  = result.sleep_stage_name
                sleep_probs = result.sleep_probs
                is_rem      = result.is_rem
                dream_cat   = result.dream_category_name
                dream_conf  = result.dream_confidence
                dream_probs_arr = result.dream_probs
            else:
                # Mock: use true hypnogram if available, else random
                if st.session_state.get("true_hypno") is not None:
                    stage = int(st.session_state["true_hypno"][i]) if i < len(st.session_state["true_hypno"]) else 0
                else:
                    stage = np.random.choice([0,1,2,3,4], p=[0.1,0.1,0.4,0.2,0.2])
                stage_name = STAGE_NAMES[stage]
                is_rem = (stage == 4)
                # Softmax-like random probs
                raw = np.random.dirichlet(np.ones(5) * 0.5)
                raw[stage] += 1.5
                sleep_probs = raw / raw.sum()
                dream_cat, dream_conf, dream_probs_arr = None, None, None

                if is_rem:
                    dream_cat_idx = np.random.randint(0, len(DREAM_CATEGORIES))
                    dream_cat  = DREAM_CATEGORIES[dream_cat_idx]
                    raw_d = np.random.dirichlet(np.ones(len(DREAM_CATEGORIES)) * 0.3)
                    raw_d[dream_cat_idx] += 1.5
                    dream_probs_arr = raw_d / raw_d.sum()
                    dream_conf = float(dream_probs_arr.max())

            sleep_timeline.append(stage)
            all_sleep_probs.append(sleep_probs)

            if is_rem and dream_cat:
                dream_predictions.append({
                    "epoch_idx":     i,
                    "timestamp_min": i * epoch_dur / 60,
                    "category_name": dream_cat,
                    "confidence":    dream_conf,
                    "probs":         dream_probs_arr,
                })

            # Update live display every epoch
            progress_bar.progress((i + 1) / n_epochs)
            status_text.markdown(
                f"**Epoch {i+1}/{n_epochs}** — "
                f"⏱ {i*epoch_dur/60:.1f} min — "
                f"Stage: **{stage_name}**"
                + (f" → 🌙 *{dream_cat}* ({dream_conf:.0%})" if is_rem and dream_cat else "")
            )

            # Live EEG plot (every 5 epochs to reduce re-renders)
            if i % 5 == 0 or is_rem:
                with live_col1:
                    eeg_chart_slot.plotly_chart(
                        plot_eeg_snippet(epoch[:min(n_ch,2)], fs, title=f"EEG — Epoch {i+1}"),
                        use_container_width=True,
                    )
                with live_col2:
                    color = STAGE_COLORS[stage_name]
                    stage_badge_slot.markdown(f"""
                    <div style="background:{color}22; border:1px solid {color}; border-radius:12px;
                                padding:16px; text-align:center; margin-top:20px;">
                        <div style="font-size:2rem;">{'🌙' if is_rem else '💤'}</div>
                        <div style="font-size:1.3rem; font-weight:bold; color:{color};">{stage_name}</div>
                        <div style="font-size:0.8rem; color:#8899bb; margin-top:4px;">
                            Confidence: {float(sleep_probs.max()):.0%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            if simulate_realtime:
                time.sleep(1.0 / sim_speed)

        st.session_state["processed"]        = True
        st.session_state["sleep_timeline"]   = sleep_timeline
        st.session_state["dream_predictions"]= dream_predictions
        st.session_state["all_sleep_probs"]  = all_sleep_probs

    # ── Results Dashboard ─────────────────────────────────────────────────────
    if st.session_state.get("processed"):
        sleep_timeline    = st.session_state["sleep_timeline"]
        dream_predictions = st.session_state["dream_predictions"]
        n_epochs          = len(sleep_timeline)
        rem_count         = sum(1 for s in sleep_timeline if s == 4)
        epoch_dur         = st.session_state.get("epoch_dur", 30)

        st.markdown("---")
        st.markdown("## 📊 Results")

        # ── Metric Cards ──────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{n_epochs}</div>
                <div class="metric-label">Total Epochs</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#ee5a24;">{rem_count}</div>
                <div class="metric-label">REM Epochs</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            rem_pct = 100 * rem_count / max(n_epochs, 1)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#c084fc;">{rem_pct:.0f}%</div>
                <div class="metric-label">REM Percentage</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#54d4b0;">{len(dream_predictions)}</div>
                <div class="metric-label">Dream Predictions</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Hypnogram ─────────────────────────────────────────────────────────
        st.markdown("### 🌙 Sleep Stage Timeline (Hypnogram)")
        st.caption("Orange regions = REM sleep (dreaming)")
        st.plotly_chart(
            plot_hypnogram(sleep_timeline, epoch_dur=epoch_dur),
            use_container_width=True,
        )

        # ── Stage Distribution + Dream Predictions ────────────────────────────
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown("### 📈 Stage Distribution")
            st.plotly_chart(plot_stage_distribution(sleep_timeline), use_container_width=True)

        with col_right:
            st.markdown("### 🌙 Dream Content Predictions")
            if dream_predictions:
                latest = dream_predictions[-1]
                st.markdown(f"**Most recent REM dream** — Epoch {latest['epoch_idx']} "
                            f"({latest['timestamp_min']:.1f} min)")
                st.plotly_chart(
                    plot_dream_probs(
                        latest["probs"],
                        DREAM_CATEGORIES[:len(latest["probs"])]
                    ),
                    use_container_width=True,
                )
            else:
                st.info("No REM sleep detected yet.")

        # ── All Dream Predictions ─────────────────────────────────────────────
        if dream_predictions:
            st.markdown("### 📋 All Dream Predictions")
            for pred in dream_predictions:
                icon = DREAM_ICONS.get(pred["category_name"], "🔵")
                conf = pred["confidence"]
                bar_filled = int(conf * 25)
                bar_empty  = 25 - bar_filled
                st.markdown(f"""
                <div class="dream-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span class="dream-title">{icon} {pred['category_name']}</span>
                            <span style="color:#556688; font-size:0.8rem; margin-left:12px;">
                                Epoch {pred['epoch_idx']} · {pred['timestamp_min']:.1f} min
                            </span>
                        </div>
                        <div style="color:#c084fc; font-weight:bold;">{conf:.0%}</div>
                    </div>
                    <div style="margin-top:8px; font-family:monospace; color:#6a5a8a; letter-spacing:1px;">
                        {'█' * bar_filled}{'░' * bar_empty}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Reset button ──────────────────────────────────────────────────────
        st.markdown("---")
        if st.button("🔄 Reset & Run Again", use_container_width=True):
            for key in ["eeg_signal", "processed", "sleep_timeline",
                        "dream_predictions", "all_sleep_probs", "true_hypno"]:
                st.session_state.pop(key, None)
            st.rerun()

    elif "eeg_signal" not in st.session_state:
        # Welcome screen
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:#556688;">
            <div style="font-size:4rem;">🧠</div>
            <h2 style="color:#3a5a8a;">Welcome to the EEG Dream Detector</h2>
            <p>Click <strong>Generate & Run Demo</strong> in the sidebar to start<br>
            — or upload your own EEG recording.</p>
            <br>
            <div style="display:inline-block; text-align:left; background:#0d1428;
                        border:1px solid #1a3060; border-radius:12px; padding:20px 30px;">
                <p style="color:#7eb8ff; margin:0 0 8px 0;"><strong>Pipeline steps:</strong></p>
                <p style="margin:4px 0; color:#8899bb;">1️⃣  Bandpass filter (1–40 Hz)</p>
                <p style="margin:4px 0; color:#8899bb;">2️⃣  DenseSleepNet → Sleep stage</p>
                <p style="margin:4px 0; color:#8899bb;">3️⃣  If REM → EEGNet → Dream content</p>
                <p style="margin:4px 0; color:#8899bb;">4️⃣  Confidence + timeline display</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()