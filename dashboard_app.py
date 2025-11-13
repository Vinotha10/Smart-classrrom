import streamlit as st
import requests
import pandas as pd
import time

SERVER = "http://127.0.0.1:5000"

st.set_page_config(layout="wide", page_title="Smart Classroom Energy Dashboard")
st.title("Smart Classroom Energy Optimization — Dashboard")

# Container for refresh
placeholder = st.empty()

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

stop_refresh = placeholder.button("Stop Refresh")

while True:
    status = get_status()
    eh = get_energy_history()

    with placeholder.container():
        left, right = st.columns([2, 1])

        # Left panel: classrooms
        with left:
            st.subheader("Classrooms")
            cols = st.columns(2)
            idx = 0
            for cls, info in status.items():
                latest = info.get('latest', {})
                pred = info.get('pred', '-')  # FIXED: use 'pred' instead of 'predicted'
                txt = f"**{cls}** — occ: {latest.get('occupancy', 0)} | temp: {latest.get('temp', 0)} °C | pred: {pred}"
                cols[idx % 2].markdown(txt)
                idx += 1

            st.markdown("---")
            st.subheader("Energy History (aggregated)")
            if not eh.empty:
                agg = eh.groupby('classroom')['total_kwh'].sum().reset_index().sort_values('total_kwh', ascending=False)
                st.dataframe(agg)
            else:
                st.write("No energy history yet.")

        # Right panel: metrics & charts
        with right:
            st.subheader("Aggregated Metrics")
            if not eh.empty:
                st.metric("Total energy (kWh)", round(eh['total_kwh'].sum(), 3))
                st.metric("Avg predicted occupancy", round(eh['predicted'].mean(), 2))
                tmp = eh.tail(100)
                st.line_chart(tmp[['predicted', 'actual']])
            else:
                st.write("No energy history yet.")

    if stop_refresh:
        break
    time.sleep(3)
