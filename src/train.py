
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split


import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ingestion")


df = pd.read_csv("data/processed/featurized_hourly_load_temp.csv", usecols =[
    'date', 'target','target_date', 'load', 'temperature','below_55', 'above_55', 'lag_1', 'lag_24', 
    'lag_168', 'hour_sin', 'hour_cos','dow_sin','dow_cos'])

df["date"] = pd.to_datetime(df["date"])
df["target_date"] = pd.to_datetime(df["target_date"])

features = ['below_55', 'above_55', 'lag_1', 'lag_24', 
    'lag_168', 'hour_sin', 'hour_cos','dow_sin','dow_cos']

X = df[features]
y = df['target']

#train model
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.3, shuffle = False)
xgb_train = xgb.DMatrix(X_train, y_train)
xgb_test = xgb.DMatrix(X_test, y_test)

#params
params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'eval_metric': 'rmse',
    'seed': 0}
n=800

model = xgb.train(
    params=params,
    dtrain=xgb_train,
    num_boost_round=500,
    evals=[(xgb_train, 'train'), (xgb_test, 'test')],
    early_stopping_rounds=30,
    verbose_eval=False)

predictions = model.predict(xgb_test)
logger.info('trained nonlinear regression')

#save results
pred = pd.DataFrame({
    "feature_date": df.loc[y_test.index, "date"],
    "target_date": df.loc[y_test.index, "target_date"],
    "actual_load": y_test,
    "predicted_load": predictions
})
pred.to_csv('data/processed/XGBoost_predictions.csv', index=False)