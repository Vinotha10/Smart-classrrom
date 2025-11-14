import streamlit as st
import requests
import pandas as pd
import time

# -------------------- CONFIG --------------------
SERVER = "http://127.0.0.1:5000"
st.set_page_config(layout="wide", page_title="Smart Classroom Energy Dashboard", page_icon="⚡")

# -------------------- THEME MODE --------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "eff_history" not in st.session_state:
    st.session_state.eff_history = []

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

dark = st.session_state.dark_mode
bg_color = "#1e1e1e" if dark else "#f7f9fb"
text_color = "#ffffff" if dark else "#222222"
card_bg = "#2e2e2e" if dark else "#ffffff"
shadow = "0 2px 6px rgba(255,255,255,0.1)" if dark else "0 2px 6px rgba(0,0,0,0.1)"

# -------------------- CUSTOM CSS --------------------
st.markdown(
    f"""
    <style>
    body {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .metric-card {{
        background-color: {card_bg};
        padding: 1rem;
        border-radius: 12px;
        box-shadow: {shadow};
        margin-bottom: 1rem;
        color: {text_color};
    }}
    .device-toggle {{
        font-size: 18px;
        margin-right: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- HEADER --------------------
col1, col2 = st.columns([8, 1])
with col1:
    st.title("🏫 Smart Classroom Energy Optimization Dashboard")
    st.caption("Real-time monitoring of classroom occupancy and energy usage using ML + rule-based control")
with col2:
    theme_label = "🌙 Dark Mode" if not dark else "☀️ Light Mode"
    st.button(theme_label, on_click=toggle_theme)

# -------------------- HELPER FUNCTIONS --------------------
def get_status():
    try:
        r = requests.get(SERVER + "/status", timeout=3)
        return r.json()
    except:
        return {}

def get_energy_history():
    try:
        r = requests.get(SERVER + "/energy_history", timeout=3)
        return pd.DataFrame(r.json())
    except:
        return pd.DataFrame()

# -------------------- LAYOUT --------------------
tab1, tab2 = st.tabs(["📊 Occupancy Overview", "⚡ Energy Efficiency"])

refresh_interval = st.sidebar.slider("⏱️ Auto-refresh interval (seconds)", 2, 10, 3)
stop_refresh = st.sidebar.checkbox("Pause Auto-Refresh", False)

placeholder = st.empty()

# -------------------- MAIN LOOP --------------------
while True:
    if stop_refresh:
        break

    status = get_status()
    eh = get_energy_history()

    with placeholder.container():
        # ---------- TAB 1: OCCUPANCY OVERVIEW ----------
        with tab1:
            st.subheader("Current Classroom Status")
            if not status:
                st.warning("No data available from server.")
            else:
                cols = st.columns(2)
                idx = 0
                for cls, info in status.items():
                    latest = info.get('latest', {})
                    pred = info.get('pred', 0)
                    occ = latest.get('occupancy', 0)
                    temp = latest.get('temp', 0)
                    motion = "✅" if latest.get('motion', 0) else "❌"

                    color = "#d1ffd1" if occ > 0 else "#ffd1d1"
                    with cols[idx % 2]:
                        st.markdown(
                            f"""
                            <div class="metric-card" style="background-color:{color}; color:{text_color}">
                                <h4>{cls.upper()}</h4>
                                <p><b>Current Occupancy:</b> {occ}</p>
                                <p><b>Predicted Next Hour:</b> {pred}</p>
                                <p><b>Temperature:</b> {temp} °C</p>
                                <p><b>Motion Detected:</b> {motion}</p>
                                <hr style="border: 1px solid {'#444' if dark else '#ccc'};">
                                <div>
                                    <span class="device-toggle">💡 Light: {'ON' if occ>0 else 'OFF'}</span>
                                    <span class="device-toggle">🌀 Fan: {'ON' if occ>0 else 'OFF'}</span>
                                    <span class="device-toggle">❄️ AC: {'ON' if temp>25 else 'OFF'}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    idx += 1

        # ---------- TAB 2: ENERGY EFFICIENCY ----------
        with tab2:
            st.subheader("Energy Usage & Source Insights")
            if not eh.empty:
                total_energy = round(eh['total_kwh'].sum(), 3)
                avg_pred = round(eh['predicted'].mean(), 2)
                solar_used = eh['use_solar'].sum()
                solar_pct = round(100 * solar_used / len(eh), 2)

                # ---- Metrics ----
                c1, c2, c3 = st.columns(3)
                c1.metric("🔋 Total Energy Used (kWh)", total_energy)
                c2.metric("👥 Avg Predicted Occupancy", avg_pred)
                c3.metric("☀️ Solar Energy Utilization", f"{solar_pct}%")

                # ---- Efficiency Calculation ----
                efficiency = 100 - (total_energy * 0.2 if total_energy < 500 else 80)
                efficiency = max(0, min(100, round(efficiency, 1)))
                st.session_state.eff_history.append(efficiency)
                if len(st.session_state.eff_history) > 50:
                    st.session_state.eff_history.pop(0)

                # ---- Animated Gauge ----
                st.markdown("### 🌡️ Energy Efficiency Indicator")
                gauge = st.progress(0, text=f"Energy Efficiency: {efficiency}%")
                for i in range(int(efficiency)):
                    time.sleep(0.005)
                    gauge.progress(i + 1, text=f"Energy Efficiency: {efficiency}%")

                # ---- Efficiency History Chart ----
                st.markdown("### 📉 Efficiency History (Last 50 readings)")
                eff_df = pd.DataFrame(st.session_state.eff_history, columns=["Efficiency (%)"])
                st.line_chart(eff_df, use_container_width=True)

                # ---- Energy Trends ----
                st.markdown("### 📈 Energy Trends")
                tmp = eh.tail(150)
                st.line_chart(tmp[['predicted', 'actual']], use_container_width=True)

                # ---- Summary ----
                st.markdown("### 🧾 Per-Classroom Summary")
                agg = eh.groupby('classroom')[['total_kwh']].sum().reset_index().sort_values('total_kwh', ascending=False)
                st.dataframe(agg, use_container_width=True)

                st.markdown("### 🔍 Latest Records")
                st.dataframe(eh.tail(10).sort_values('timestamp', ascending=False), use_container_width=True)
            else:
                st.info("Energy data not available yet. Run the simulator to start streaming data.")

    time.sleep(refresh_interval)
