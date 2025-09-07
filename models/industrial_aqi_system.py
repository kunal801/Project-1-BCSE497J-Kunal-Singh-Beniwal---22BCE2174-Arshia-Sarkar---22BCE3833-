# industrial_aqi_system.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class IndustrialAQIModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.target_gases = ['CO', 'CO2', 'NH3', 'H2S', 'VOC', 'PM2_5']
        self.window_size = 12  # Past 1 hour at 5-min intervals
        self.thresholds = {
            'CO':   {'safe': 9,     'caution': 25,    'danger': 35},
            'CO2':  {'safe': 1000,  'caution': 5000,  'danger': 40000},
            'NH3':  {'safe': 5,     'caution': 15,    'danger': 25},
            'H2S':  {'safe': 10,    'caution': 15,    'danger': 20},
            'VOC':  {'safe': 30,    'caution': 50,    'danger': 100},
            'PM2_5':{'safe': 35,    'caution': 55,    'danger': 150}
        }

    def generate_sample_data(self, start_date='2025-01-01', days=90):
        date_range = pd.date_range(start=start_date, periods=days*24*12, freq='5T')
        n = len(date_range)
        np.random.seed(42)
        data = pd.DataFrame({
            'timestamp': date_range,
            'CO': np.clip(np.random.normal(8,2,n)+3*np.sin(np.arange(n)*0.01),0,50),
            'CO2': np.clip(np.random.normal(450,50,n)+100*np.sin(np.arange(n)*0.005),300,800),
            'NH3': np.clip(np.random.normal(2,0.5,n)+np.sin(np.arange(n)*0.02),0,10),
            'H2S': np.clip(np.random.normal(0.5,0.2,n)+0.3*np.sin(np.arange(n)*0.015),0,5),
            'VOC': np.clip(np.random.normal(15,5,n)+10*np.sin(np.arange(n)*0.008),0,100),
            'PM2_5': np.clip(np.random.normal(25,8,n)+15*np.sin(np.arange(n)*0.006),0,100),
            'temperature': 25 + 10*np.sin(np.arange(n)*0.0001) + np.random.normal(0,2,n),
            'humidity': np.clip(60 + 20*np.sin(np.arange(n)*0.0002) + np.random.normal(0,5,n),20,90),
            'wind_speed': np.abs(np.random.normal(3,1.5,n)),
            'production_activity': np.random.choice([0,1],n,p=[0.3,0.7])
        })
        hrs = data['timestamp'].dt.hour
        mask = (hrs>=8)&(hrs<=18)
        data.loc[mask, ['CO','CO2','VOC','PM2_5']] *= [1.3,1.2,1.5,1.4]
        data.loc[data['production_activity']==1, ['CO','NH3','H2S']] *= [1.2,1.4,1.3]
        return data

    def _engineer_features(self, df):
        df2 = df.copy()
        df2['hour'] = df2['timestamp'].dt.hour
        df2['day'] = df2['timestamp'].dt.dayofweek
        df2['is_work'] = df2['hour'].between(8,18).astype(int)
        for gas in self.target_gases:
            df2[f'{gas}_rate'] = df2[gas].diff().fillna(0)
            df2[f'{gas}_ma']   = df2[gas].rolling(self.window_size).mean().fillna(df2[gas])
        df2['temp_hum']  = df2['temperature'] * df2['humidity']
        df2['wind_temp'] = df2['wind_speed'] * df2['temperature']
        return df2

    def train(self, df):
        df_feat = self._engineer_features(df)
        for gas in self.target_gases:
            features = ['temperature','humidity','wind_speed','production_activity',
                        'hour','day','is_work','temp_hum','wind_temp'] + \
                       [g for g in self.target_gases if g!=gas] + \
                       [f'{gas}_rate', f'{gas}_ma']
            X = df_feat[features].fillna(0)
            y = df_feat[gas]
            scaler = MinMaxScaler().fit(X)
            Xs = scaler.transform(X)
            split = int(len(Xs)*0.8)
            Xtr, Xte = Xs[:split], Xs[split:]
            ytr, yte = y[:split], y[split:]
            model = RandomForestRegressor(n_estimators=100,
                                          max_depth=15,
                                          random_state=42,
                                          n_jobs=-1)
            model.fit(Xtr, ytr)
            self.scalers[gas] = (scaler, features)
            self.models[gas]  = model
            ypr = model.predict(Xte)
            r2  = np.corrcoef(yte, ypr)[0,1]**2 if len(yte)>1 else float('nan')
            rmse = np.sqrt(mean_squared_error(yte, ypr))
            print(f"{gas}: R²={r2:.3f}, RMSE={rmse:.3f}")

    def _assess(self, gas, val, thr):
        if val <= thr['safe']:
            return 'SAFE'
        if val <= thr['caution']:
            return 'CAUTION'
        return 'DANGER'

    def calculate_overall_aqi(self, preds):
        danger  = sum(1 for v in preds.values() if v['lvl']=='DANGER')
        caution = sum(1 for v in preds.values() if v['lvl']=='CAUTION')
        if danger > 0:
            return 'HAZARDOUS'
        if caution >= 2:
            return 'UNHEALTHY'
        if caution == 1:
            return 'MODERATE'
        return 'GOOD'

    def predict(self, df, dt):
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        df2 = df.copy()
        df2['diff'] = abs(df2['timestamp'] - dt)
        idx = df2['diff'].idxmin()
        if idx < self.window_size:
            return None
        ctx = df2.iloc[idx-self.window_size:idx+1]
        ctxf= self._engineer_features(ctx).iloc[-1]
        preds = {}
        for gas in self.target_gases:
            scaler, feat = self.scalers[gas]
            X = scaler.transform([ctxf[feat].fillna(0).values])
            pv = max(0, self.models[gas].predict(X)[0])
            lvl= self._assess(gas, pv, self.thresholds[gas])
            preds[gas] = {'pred': round(pv,2), 'lvl': lvl}
        overall = self.calculate_overall_aqi(preds)
        return {'datetime': dt, 'predictions': preds, 'overall_aqi': overall}

    def save(self, path='aqi_model.pkl'):
        joblib.dump({'models': self.models, 'scalers': self.scalers}, path)

    def load(self, path='aqi_model.pkl'):
        data = joblib.load(path)
        self.models, self.scalers = data['models'], data['scalers']

def main():
    model = IndustrialAQIModel()
    data  = model.generate_sample_data()
    print("Generated sample data.")
    model.train(data)
    model.save()
    # Demo predictions
    for ts in ['2025-02-15 10:30:00', '2025-02-20 02:15:00']:
        out = model.predict(data, ts)
        print(f"\nPrediction for {ts}: Overall AQI = {out['overall_aqi']}")
        for gas, info in out['predictions'].items():
            print(f"  {gas}: {info['pred']} ({info['lvl']})")

if __name__ == '__main__':
    main()
