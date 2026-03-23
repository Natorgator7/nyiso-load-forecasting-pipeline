
import pandas as pd
import numpy as np
from ingest import ingest_csv
import logging

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ingestion")

df = ingest_csv('data/processed/hourly_load_temp.csv', 
                ['date','load', 'temperature'])
t = df['temperature']
df['date'] = pd.to_datetime(df['date'], errors='coerce')


#implement cooling, heating, hour lag, day lag, week lag 
df['below_55'] = (55 - t).clip(lower=0)
df['above_55'] = (t - 55).clip(lower=0)
df['lag_1'] = df['load'].shift(1)
df['lag_24'] = df['load'].shift(24)
df['lag_168'] = df['load'].shift(168)
df['target'] = df['load'].shift(-24)

df = df.dropna()



#implement periodic time correlates (hour, day of week)
df['hour'] = df['date'].dt.hour
df['dow'] = df['date'].dt.dayofweek
df['doy'] = df['date'].dt.dayofyear

df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
df['dow_sin'] = np.sin(2*np.pi*df['dow']/7)
df['dow_cos'] = np.cos(2*np.pi*df['dow']/7)

#true forecast target: 24 hours ahead
df['target'] = df['load'].shift(-24)
df['target_date'] = df["date"].shift(-24)

#drop rows with incomplete feature/target data
df = df.dropna().reset_index(drop=True)
#save features
df.to_csv("data/processed/featurized_hourly_load_temp.csv",index=False)

#run multivariate linear regression for baseline
X = df[['below_55', 'above_55', 'lag_1', 'lag_24', 
        'lag_168', 'hour_sin', 'hour_cos','dow_sin','dow_cos']]
y = df['target']

logger.info('featurized data with heating hours, cooling hours,'\
'\n hour, day, and week lags, and periodic data')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=0.3)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
naive_pred = X_test['lag_24']
logger.info('trained baseline linear regression')

#save results
pred_series = pd.DataFrame({
    "feature_date": df.loc[y_test.index, "date"],
    "target_date": df.loc[y_test.index, "target_date"],
    "actual_load": y_test,
    "predicted_load": predictions,
    "naive_pred": naive_pred,
})
pred_series.to_csv('data/processed/baseline_predictions.csv', index=False)
# %%
