# model_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from control import rule_based_control
from tensorflow.keras.models import load_model

app = Flask("smart_brain")
CORS(app)

# ------------------ Paths ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ------------------ Load Models ------------------
lstm, scaler_lstm, rf, scaler_rf = None, None, None, None

# --- Load LSTM ---
lstm_path = os.path.join(MODELS_DIR, "lstm_occ.h5")
scaler_lstm_path = os.path.join(MODELS_DIR, "scaler_lstm.joblib")

if os.path.exists(lstm_path) and os.path.exists(scaler_lstm_path):
    try:
        lstm = load_model(lstm_path)
        scaler_lstm = joblib.load(scaler_lstm_path)
        print("✅ LSTM model and scaler loaded successfully.")
    except Exception as e:
        print(f"⚠️ Error loading LSTM or scaler: {e}")
else:
    print("⚠️ LSTM model or scaler not found in:", MODELS_DIR)

# --- Load Random Forest ---
rf_path = os.path.join(MODELS_DIR, "rf_model.joblib")
scaler_rf_path = os.path.join(MODELS_DIR, "scaler_rf.joblib")

if os.path.exists(rf_path) and os.path.exists(scaler_rf_path):
    try:
        rf = joblib.load(rf_path)
        scaler_rf = joblib.load(scaler_rf_path)
        print("✅ Random Forest model and scaler loaded successfully.")
    except Exception as e:
        print(f"⚠️ Error loading Random Forest or scaler: {e}")
else:
    print("⚠️ Random Forest model or scaler not found in:", MODELS_DIR)

# ------------------ In-memory state ------------------
LATEST = {}          # {classroom: {...}}
ENERGY_HISTORY = []  # list of dicts for dashboard plots


# ------------------ Helper: LSTM preprocessing ------------------
def preprocess_seq_for_lstm(classroom):
    """Build last 6 timesteps for LSTM from stored history."""
    if classroom not in LATEST or 'history' not in LATEST[classroom]:
        return None

    hist = LATEST[classroom]['history']
    if len(hist) < 6:
        return None  # need at least 6 past steps

    cols = ['hour', 'dow', 'is_holiday', 'scheduled', 'occ_lag1',
            'motion', 'temp', 'co2', 'solar_kw']
    arr = [[r.get(c, 0) for c in cols] for r in hist[-6:]]
    seq = np.array(arr)

    if scaler_lstm:
        flat = seq.reshape(-1, seq.shape[-1])
        flat_s = scaler_lstm.transform(flat)
        seq_s = flat_s.reshape(1, seq.shape[0], seq.shape[1])
        return seq_s
    else:
        return seq.reshape(1, seq.shape[0], seq.shape[1])


# ------------------ Helper: RF preprocessing ------------------
def make_rf_features(rec):
    """Extract feature vector for Random Forest prediction."""
    cols = ['hour', 'dow', 'is_holiday', 'scheduled', 'occ_lag1',
            'motion', 'temp', 'co2', 'solar_kw']
    return np.array([rec.get(c, 0) for c in cols]).reshape(1, -1)


# ------------------ Routes ------------------
@app.route("/update", methods=["POST"])
def update():
    j = request.get_json()
    cls = j['classroom']

    # Initialize classroom data if not already
    if cls not in LATEST:
        LATEST[cls] = {"history": []}

    rec = j.copy()
    ts = pd.to_datetime(j['timestamp'])
    rec['hour'] = ts.hour
    rec['dow'] = ts.weekday()
    rec['occ_lag1'] = LATEST[cls]['history'][-1]['occupancy'] if LATEST[cls]['history'] else 0

    # Append to classroom history
    LATEST[cls]['history'].append(rec)
    LATEST[cls]['history'] = LATEST[cls]['history'][-48:]  # keep last 48 entries (~1 day if 30-min intervals)
    LATEST[cls]['latest'] = rec
    LATEST[cls]['last_update'] = datetime.utcnow().isoformat()

    # ---------------- Prediction ----------------
    pred = None

    try:
        # Prefer Random Forest if available
        if rf is not None:
            X = make_rf_features(rec)
            Xs = scaler_rf.transform(X) if scaler_rf else X
            pred_val = rf.predict(Xs)[0]
            pred = max(0, int(round(pred_val)))
            print(f"[{cls}] ✅ RF prediction = {pred} (raw={pred_val:.2f})")

        # Else fallback to LSTM if available
        elif lstm is not None:
            seq = preprocess_seq_for_lstm(cls)
            if seq is not None:
                p = lstm.predict(seq)[0][0]
                pred = max(0, int(round(p)))
                print(f"[{cls}] ✅ LSTM prediction = {pred}")
            else:
                print(f"[{cls}] ⚠️ Not enough history for LSTM (need 6 timesteps).")
                pred = 0

        else:
            print(f"[{cls}] ⚠️ No model available. Returning 0.")
            pred = 0

    except Exception as e:
        print(f"[{cls}] ❌ Prediction error: {e}")
        pred = 0

    # ---------------- Control Logic ----------------
    ctr = rule_based_control(rec, pred)

    # ---------------- Energy Logging ----------------
    ENERGY_HISTORY.append({
        "timestamp": rec['timestamp'],
        "classroom": cls,
        "predicted": pred,
        "actual": rec['occupancy'],
        **ctr['energy'],
        "use_solar": ctr['use_solar']
    })

    return jsonify({
        "predicted_occupancy": pred,
        "control": ctr['devices'],
        "energy": ctr['energy'],
        "use_solar": ctr['use_solar']
    })


@app.route("/status", methods=["GET"])
def status():
    result = {}
    for cls, data in LATEST.items():
        latest = data.get("latest", {})
        pred = data.get("pred", 0)  # <- ensure prediction key exists
        result[cls] = {
            "latest": latest,
            "pred": pred
        }
    return jsonify(result)

@app.route("/energy_history", methods=["GET"])
def energy_hist():
    """Return last 200 energy history records."""
    return jsonify(ENERGY_HISTORY[-200:])


# ------------------ Main ------------------
if __name__ == "__main__":
    print("\n🔹 Smart Classroom Model Server Started on port 5000 🔹")
    app.run(port=5000, debug=True)
