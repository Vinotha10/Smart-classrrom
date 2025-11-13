# train_model.py
import os, argparse, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

DATA_FN = "data/sim_data.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def prepare(df):
    # convert timestamp and create time features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['classroom','timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.weekday
    df['occ_lag1'] = df.groupby('classroom')['occupancy'].shift(1).fillna(0)
    # choose features
    cols = ['hour','dow','is_holiday','scheduled','occ_lag1','motion','temp','co2','solar_kw']
    df = df.fillna(0)
    return df, cols

def train_rf(df, cols):
    X = df[cols].values
    y = df['occupancy'].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_s, y_train)
    print("RF score:", rf.score(X_test_s, y_test))
    joblib.dump(rf, os.path.join(MODELS_DIR, "rf_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_rf.joblib"))

def create_sequences(df, cols, seq_len=6):
    # returns X, y for LSTM: sequences of last seq_len timesteps to predict next occupancy
    Xs, ys = [], []
    grouped = df.groupby('classroom')
    for name, g in grouped:
        g = g.sort_values('timestamp')
        vals = g[cols].values
        occ = g['occupancy'].values
        for i in range(len(g)-seq_len):
            Xs.append(vals[i:i+seq_len])
            ys.append(occ[i+seq_len])  # next timestep occupancy
    return np.array(Xs), np.array(ys)

def train_lstm(df, cols):
    X, y = create_sequences(df, cols, seq_len=6)
    # scale features per feature dimension
    n_samples, seq_len, n_feats = X.shape
    X_flat = X.reshape(n_samples*seq_len, n_feats)
    scaler = StandardScaler().fit(X_flat)
    Xs = scaler.transform(X_flat).reshape(n_samples, seq_len, n_feats)
    # train-test split
    train_n = int(n_samples*0.8)
    X_train, X_test = Xs[:train_n], Xs[train_n:]
    y_train, y_test = y[:train_n], y[train_n:]
    model = Sequential([
        LSTM(64, input_shape=(seq_len, n_feats), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=50, batch_size=64, callbacks=[es])
    # evaluate
    loss = model.evaluate(X_test,y_test)
    print("Test loss:", loss)
    model.save(os.path.join(MODELS_DIR, "lstm_occ.h5"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_lstm.joblib"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_rf", action="store_true")
    args = parser.parse_args()
    df = pd.read_csv(DATA_FN)
    df, cols = prepare(df)
    if args.use_rf:
        train_rf(df, cols)
    else:
        train_lstm(df, cols)
