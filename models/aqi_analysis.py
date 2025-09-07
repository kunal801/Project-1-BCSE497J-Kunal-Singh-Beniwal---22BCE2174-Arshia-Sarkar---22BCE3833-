import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# 1. Load and concatenate all CSVs in the 'data' folder
csv_files = glob.glob(os.path.join('data', '*.csv'))
df_list = [pd.read_csv(f, parse_dates=['timestamp']) for f in csv_files]
data = pd.concat(df_list, ignore_index=True)
print(f"Loaded {len(csv_files)} files, {data.shape[0]} rows total.")

# 2. Feature engineering
data['hour'] = data['timestamp'].dt.hour
data['dayofweek'] = data['timestamp'].dt.dayofweek
data['temp_hum'] = data['temperature'] * data['humidity']
data['AQI'] = data[['CO', 'CO2', 'NH3', 'H2S', 'VOC', 'PM2_5']].max(axis=1)  # surrogate overall AQI

# 3. Binary classification target: Safe vs Hazardous (AQI>150)
data['is_hazard'] = (data['AQI'] > 150).astype(int)

features = ['CO','CO2','NH3','H2S','VOC','PM2_5','temperature','humidity','wind_speed','hour','dayofweek','temp_hum']
X = data[features]
y_reg = data['AQI']
y_clf = data['is_hazard']

# 4. Train/Test split
X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)

# 5. Regression model
reg = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
reg.fit(X_train, y_train_reg)
y_pred_reg = reg.predict(X_test)

# Regression metrics
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)
print("Regression Performance:")
print(f"  MSE: {mse:.2f}")
print(f"  RMSE: {rmse:.2f}")
print(f"  R²: {r2:.3f}")

# Plot actual vs predicted AQI
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test_reg, y=y_pred_reg, alpha=0.5)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.tight_layout()
plt.savefig("aqi_scatter.png")
plt.close()

# 6. Classification from regression output
y_pred_clf = (y_pred_reg > 150).astype(int)

# Classification metrics
acc = accuracy_score(y_test_clf, y_pred_clf)
prec = precision_score(y_test_clf, y_pred_clf)
rec = recall_score(y_test_clf, y_pred_clf)
f1 = f1_score(y_test_clf, y_pred_clf)
print("\nClassification Performance (AQI>150):")
print(f"  Accuracy: {acc:.3f}")
print(f"  Precision: {prec:.3f}")
print(f"  Recall: {rec:.3f}")
print(f"  F1 Score: {f1:.3f}")

# Confusion matrix heatmap
cm = confusion_matrix(y_test_clf, y_pred_clf)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Safe','Hazardous'], yticklabels=['Safe','Hazardous'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# 7. Save regression model
import joblib
joblib.dump(reg, "aqi_reg_model.pkl")

print("\nPlots saved: aqi_scatter.png, confusion_matrix.png")
print("Model saved: aqi_reg_model.pkl")
