import streamlit as st
import requests
import pandas as pd
import time
import streamlit.components.v1 as components
import os

# ----------------------------
# Configuration
# ----------------------------
SERVER = "http://127.0.0.1:5000"   # backend

st.set_page_config(layout="wide", page_title="Smart Classroom Energy Dashboard", page_icon="⚡")

# simple CSS to ensure readable text & card style
st.markdown(
    """
    <style>
    .metric-card {
        background-color: white;
        padding: 0.9rem;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        margin-bottom: 0.9rem;
        color: #111;
    }
    .stMarkdown p, .stMarkdown h4 { color: #111 !important; }
    .small-muted { color: #555; font-size:13px; }
    /* make embedded html overlay readable when embedded in dark theme */
    .stHtmlContainer { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏫 Smart Classroom Energy Optimization Dashboard")
st.caption("Real-time monitoring of classroom occupancy and energy usage using ML + rule-based control")

# ---------- Helper Functions ----------
def get_status():
    try:
        r = requests.get(SERVER + "/status", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def get_energy_history():
    try:
        r = requests.get(SERVER + "/energy_history", timeout=4)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame()

# ---------- Layout ----------
tab1, tab2, tab3 = st.tabs(["📊 Occupancy Overview", "⚡ Energy Efficiency", "🧭 3D Classroom"])

refresh_interval = st.sidebar.slider("⏱️ Auto-refresh interval (seconds)", 2, 10, 3)
pause = st.sidebar.checkbox("Pause Auto-Refresh", False)
# path to the 3D html file (resolve relative to this script)
HTML_3D_FN = os.path.join(os.path.dirname(__file__), "classroom_3d.html")

# Use session_state for a safe auto-refresh cycle (avoids deprecated query param APIs)
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = 0

now_ts = int(time.time())
should_rerun = (not pause) and (now_ts - st.session_state["last_refresh"] >= refresh_interval)

# Fetch data once per run
status = get_status()
eh = get_energy_history()

# ---------- TAB 1: OCCUPANCY OVERVIEW ----------
with tab1:
    st.subheader("Current Classroom Status")
    if not status:
        st.warning("No data available from server. Start the backend (model_server) and run simulator.")
    else:
        cols = st.columns(2)
        idx = 0
        # sort keys for stable ordering
        for cls in sorted(status.keys()):
            info = status[cls]
            latest = info.get('latest', {})
            # older server used 'pred' key; fallback to predicted_occupancy or latest occupancy
            pred = info.get('pred', None)
            if pred is None:
                pred = latest.get('predicted_occupancy', None)
            if pred is None:
                pred = latest.get('occupancy', '-')
            occ = latest.get('occupancy', 0)
            temp = latest.get('temp', 0)
            motion = "✅" if latest.get('motion', 0) else "❌"

            color = "#e6ffe6" if occ > 0 else "#ffe6e6"
            with cols[idx % 2]:
                st.markdown(
                    f"""
                    <div class="metric-card" style="background-color:{color}; color:#111;">
                        <h4 style="margin:0 0 6px 0">{cls.upper()}</h4>
                        <p style="margin:0"><b>Current Occupancy:</b> {occ}</p>
                        <p style="margin:0"><b>Predicted Next Hour:</b> {pred}</p>
                        <p style="margin:0"><b>Temperature:</b> {temp} °C</p>
                        <p style="margin:0"><b>Motion Detected:</b> {motion}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            idx += 1

# ---------- TAB 2: ENERGY EFFICIENCY ----------
with tab2:
    st.subheader("Energy Usage & Source Insights")
    if not eh.empty:
        # defensive checks
        if 'total_kwh' not in eh.columns:
            st.error("Energy history has unexpected format (missing total_kwh).")
        else:
            total_energy = round(eh['total_kwh'].sum(), 3)
            avg_pred = round(eh['predicted'].mean(), 2) if 'predicted' in eh.columns else 0.0
            try:
                solar_used = eh['use_solar'].astype(int).sum() if 'use_solar' in eh.columns else 0
                solar_pct = round(100 * solar_used / len(eh), 2) if len(eh) else 0.0
            except Exception:
                solar_pct = 0.0

            c1, c2, c3 = st.columns(3)
            c1.metric("🔋 Total Energy Used (kWh)", total_energy)
            c2.metric("👥 Avg Predicted Occupancy", avg_pred)
            c3.metric("☀️ Solar Energy Utilization", f"{solar_pct}%")

            st.markdown("### 📈 Energy Trends")
            tmp = eh.tail(150)
            try:
                st.line_chart(tmp[['predicted', 'actual']], use_container_width=True)
            except Exception:
                st.write("Chart unavailable: check energy history shape.")

            st.markdown("### 🧾 Per-Classroom Summary")
            agg = eh.groupby('classroom')[['total_kwh']].sum().reset_index().sort_values('total_kwh', ascending=False)
            st.dataframe(agg, use_container_width=True)

            st.markdown("### 🔍 Latest Records")
            st.dataframe(eh.tail(10).sort_values('timestamp', ascending=False), use_container_width=True)
    else:
        st.info("Energy data not available yet. Run the simulator to start streaming data.")

# ---------- TAB 3: 3D Classroom ----------
with tab3:
    st.subheader("Interactive 3D Classroom (Simple block-based)")
    st.markdown("The 3D view polls the backend and derives device states locally (lights/fan/AC) from the predicted occupancy so no server changes are needed.")
    if os.path.exists(HTML_3D_FN):
        html_str = open(HTML_3D_FN, "r", encoding="utf-8").read()
        # height sized for most monitors; adjust as needed
        components.html(html_str, height=680, scrolling=False)
    else:
        st.error(f"3D file not found: {HTML_3D_FN}. Place `classroom_3d.html` next to this script.")

# ---------- Auto-refresh control (session_state based) ----------
if should_rerun:
    # update last_refresh and request rerun (non-blocking)
    st.session_state["last_refresh"] = now_ts
    # new function to request rerun (older experimental_rerun can be removed in some versions)
    try:
        st.experimental_request_rerun()
    except Exception:
        # fallback: try to use the older name if available (keeps compatibility)
        try:
            st.experimental_rerun()
        except Exception:
            # if neither exists, gracefully continue (no hard failure)
            pass
else:
    # sidebar footer info
    st.sidebar.markdown(f"Last poll: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now_ts))}")
    if pause:
        st.sidebar.info("Auto-refresh is paused.")
