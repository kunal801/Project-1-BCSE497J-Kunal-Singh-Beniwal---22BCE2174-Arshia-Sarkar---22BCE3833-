import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings("ignore")

# -------------- 1) Synthesise dataset --------------
np.random.seed(42)

n_minutes = 60*24  # one day of minute-resolution data
time_idx = pd.date_range(start="2025-09-01", periods=n_minutes, freq="T")

true_conc = 20 + 5*np.sin(np.linspace(0, 4*math.pi, n_minutes))
true_conc += np.where((time_idx.hour>=9) & (time_idx.hour<=17), 5, 0)
true_conc += np.random.normal(0, 0.8, size=n_minutes)

temp = 22 + 3*np.sin(np.linspace(0, 2*math.pi, n_minutes)) + np.random.normal(0,0.3,n_minutes)
rh = 45 + 10*np.sin(np.linspace(0, 2*math.pi, n_minutes)+1.0) + np.random.normal(0,1.0,n_minutes)

s1 = 0.8*true_conc + 0.05*temp - 0.02*rh + np.random.normal(0,1.2,n_minutes) + 0.01*np.arange(n_minutes)/n_minutes*50
s2 = 1.1*true_conc - 0.03*temp + 0.03*rh + np.random.normal(0,1.5,n_minutes) + 0.02*np.sin(np.linspace(0,6*math.pi,n_minutes))
s3 = 0.5*true_conc + 0.1*temp + np.random.normal(0,0.9,n_minutes)

df = pd.DataFrame({
    "time": time_idx,
    "true_conc": true_conc,
    "sensor_1": s1,
    "sensor_2": s2,
    "sensor_3": s3,
    "temp": temp,
    "rh": rh
})
df.set_index("time", inplace=True)

# Add hazardous spike
df_spike = df.copy()
spike_start = pd.Timestamp("2025-09-01 15:40")
df_spike.loc[spike_start:spike_start+pd.Timedelta(minutes=15), "true_conc"] += np.linspace(0, 40, 16)
df_spike.loc[spike_start:spike_start+pd.Timedelta(minutes=15), "sensor_1"] += np.linspace(0, 35, 16)
df_spike.loc[spike_start:spike_start+pd.Timedelta(minutes=15), "sensor_2"] += np.linspace(0, 45, 16)
df_spike.loc[spike_start:spike_start+pd.Timedelta(minutes=15), "sensor_3"] += np.linspace(0, 20, 16)
df = df_spike

# -------------- 2) Calibration model --------------
def create_features(data, lags=[1,2,3,5,10]):
    X = data[["sensor_1","sensor_2","sensor_3","temp","rh"]].copy()
    for lag in lags:
        X[f"s1_lag_{lag}"] = data["sensor_1"].shift(lag)
        X[f"s2_lag_{lag}"] = data["sensor_2"].shift(lag)
        X[f"s3_lag_{lag}"] = data["sensor_3"].shift(lag)
    X["s1_roll_mean_5"] = data["sensor_1"].rolling(5, min_periods=1).mean()
    X["s2_roll_std_5"] = data["sensor_2"].rolling(5, min_periods=1).std().fillna(0)
    X["s3_roll_mean_10"] = data["sensor_3"].rolling(10, min_periods=1).mean()
    X = X.fillna(method="bfill").fillna(method="ffill")
    return X

X = create_features(df)
y = df["true_conc"]

split_idx = int(len(df)*0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

calib = RandomForestRegressor(n_estimators=100, random_state=42)
calib.fit(X_train, y_train)
y_pred_calib = calib.predict(X_test)

# FIX: manual RMSE
mae = mean_absolute_error(y_test, y_pred_calib)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_calib))
print("Calibration MAE:", mae)
print("Calibration RMSE:", rmse)

# -------------- 3) Forecasting model --------------
df["est_conc"] = calib.predict(X)

def create_forecast_features(est_series, lags=[1,2,3,5,10,30]):
    F = pd.DataFrame({"est": est_series})
    for lag in lags:
        F[f"est_lag_{lag}"] = est_series.shift(lag)
    F = F.fillna(method="bfill").fillna(method="ffill")
    return F

H = create_forecast_features(df["est_conc"])
H_train, H_test = H.iloc[:split_idx], H.iloc[split_idx:]
y_f_train = df["true_conc"].shift(-1).fillna(method="ffill").iloc[:split_idx]
y_f_test = df["true_conc"].shift(-1).fillna(method="ffill").iloc[split_idx:]

forecast_model = RandomForestRegressor(n_estimators=100, random_state=1)
forecast_model.fit(H_train, y_f_train)

def forecast_recursive(last_known_df, steps=30):
    preds = []
    curr = last_known_df.copy().iloc[-1:].copy()
    for i in range(steps):
        p = forecast_model.predict(curr)[0]
        preds.append(p)
        new = {}
        new["est"] = p
        for col in curr.columns:
            if col.startswith("est_lag_"):
                lag = int(col.split("_")[-1])
                if lag == 1:
                    new[col] = curr["est"].values[0]
                else:
                    prev_lag_col = f"est_lag_{lag-1}"
                    new[col] = curr[prev_lag_col].values[0] if prev_lag_col in curr.columns else curr["est"].values[0]
        curr = pd.DataFrame(new, index=[0])
    return np.array(preds)

last_window = H_test.tail(1)
fcst = forecast_recursive(last_window, steps=30)

y1_pred = forecast_model.predict(H_test)
mae_f = mean_absolute_error(y_f_test, y1_pred)
rmse_f = np.sqrt(mean_squared_error(y_f_test, y1_pred))
print("1-step Forecast MAE:", mae_f)
print("1-step Forecast RMSE:", rmse_f)

# -------------- 4) Anomaly detection --------------
anom_features = df[["sensor_1","sensor_2","sensor_3","temp","rh","est_conc"]].copy()
iso = IsolationForest(contamination=0.001, random_state=42)
iso.fit(anom_features.iloc[:split_idx])
scores = iso.decision_function(anom_features)
preds_anom = iso.predict(anom_features)

df["anomaly_score"] = scores
df["anomaly_flag"] = (preds_anom == -1).astype(int)
print("Number of anomalies detected (whole day):", df["anomaly_flag"].sum())

# -------------- 5) Risk scoring --------------
OEL = 50.0
def risk_level(conc, oel=OEL):
    if conc >= oel:
        return "DANGER", 2
    elif conc >= 0.5*oel:
        return "CAUTION", 1
    else:
        return "SAFE", 0

df["risk_level"] = df["est_conc"].apply(lambda x: risk_level(x)[0])
df["risk_code"] = df["est_conc"].apply(lambda x: risk_level(x)[1])

# -------------- 6) Explainability --------------
perm = permutation_importance(calib, X_test, y_test, n_repeats=10, random_state=0)
importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
}).sort_values("importance_mean", ascending=False).reset_index(drop=True)

# -------------- 7) Outputs --------------
window = df.loc["2025-09-01 15:20":"2025-09-01 16:10"]

plt.figure(figsize=(10,4))
plt.plot(window.index, window["true_conc"], label="true_conc")
plt.plot(window.index, window["est_conc"], label="est_conc", linestyle="--")
plt.title("True vs Calibrated Estimate (window with spike)")
plt.xlabel("time")
plt.ylabel("concentration")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,3))
plt.plot(df.index, df["anomaly_score"])
plt.title("Anomaly Score (whole day)")
plt.xlabel("time")
plt.tight_layout()
plt.show()

print("\nTop features for calibration model (by permutation importance):")
print(importance_df.head(10).to_string(index=False))

print("\nExample 30-min recursive forecast (first 10 values):")
print(np.round(fcst[:10], 2))

alerts = df[(df["risk_code"] > 0) | (df["anomaly_flag"] == 1)]
print(f"\nFound {len(alerts)} alert-worthy rows. Showing first 8:")
print(alerts.head(8)[["est_conc", "risk_level", "anomaly_flag"]].to_string())
