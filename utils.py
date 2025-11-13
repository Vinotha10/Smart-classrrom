# utils.py
import numpy as np

def scale_features(arr, scaler=None):
    """
    Scale input array using a scaler if provided.
    If no scaler, return original array.
    """
    if scaler:
        return scaler.transform(arr)
    return arr

def calculate_energy(devices):
    """
    Calculate energy consumption based on devices dictionary.
    Example input:
        devices = {"ac": 1, "fan":1, "lights":1}
    Example output:
        {"ac_kwh": 0.5, "fan_kwh": 0.1, "lights_kwh":0.2, "total_kwh":0.8}
    """
    # Example power ratings (kW per hour)
    power = {"ac": 0.5, "fan": 0.1, "lights": 0.2}
    energy = {}
    total = 0.0
    for k,v in devices.items():
        if k in power:
            energy_kwh = power[k] * v  # v=0 or 1
            energy[k + "_kwh"] = round(energy_kwh, 3)
            total += energy_kwh
    energy["total_kwh"] = round(total,3)
    return energy

def prepare_rf_features(state):
    """
    Convert sensor state dict into RF feature array
    """
    cols = ['hour','dow','is_holiday','scheduled','occ_lag1','motion','temp','co2','solar_kw']
    return np.array([state.get(c,0) for c in cols]).reshape(1,-1)

def prepare_lstm_sequence(history, scaler=None):
    """
    Convert last 6 timesteps into LSTM input
    """
    if len(history) < 6:
        return None
    cols = ['hour','dow','is_holiday','scheduled','occ_lag1','motion','temp','co2','solar_kw']
    arr = np.array([[r.get(c,0) for c in cols] for r in history[-6:]])
    if scaler:
        flat = arr.reshape(-1, arr.shape[-1])
        flat_s = scaler.transform(flat)
        arr_s = flat_s.reshape(1, arr.shape[0], arr.shape[1])
        return arr_s
    return arr.reshape(1, arr.shape[0], arr.shape[1])
