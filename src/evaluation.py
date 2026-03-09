
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ingestion")

from ingest import ingest_csv

#import model outputs
data = ingest_csv('data/processed/featurized_hourly_load_temp.csv', 
                  ['date', 'load'])
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

baseline = ingest_csv('data/processed/baseline_predictions.csv', 
                    ['date', 'predicted_load', 'naive_pred'])
baseline['date'] = pd.to_datetime(baseline['date'])
baseline.set_index('date', inplace=True)

xgb = ingest_csv('data/processed/XGBoost_predictions.csv', 
                    ['date', 'predicted_load'])
xgb['date'] = pd.to_datetime(xgb['date'])
xgb.set_index('date', inplace=True)

load_aligned = data['load'].reindex(xgb.index)

#graphing
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

# Subplot 1: actual vs baseline
data['load'].plot(ax=ax1, label='Actual Load', alpha=0.7)
baseline['predicted_load'].plot(ax=ax1, label='Baseline Pred', alpha=0.6)
ax1.set_title('Actual vs Baseline Prediction')
ax1.set_xlabel('Date')
ax1.set_ylabel('Load (MW)')
ax1.legend()

# Subplot 2: actual vs XGBoost
data['load'].plot(ax=ax2, label='Actual Load', alpha=0.7)
xgb['predicted_load'].plot(ax=ax2, label='XGBoost Pred', alpha=0.6, color = 'green')
ax2.set_title('Actual vs XGBoost Prediction')
ax2.set_xlabel('Date')
ax2.set_ylabel('Load (MW)')
ax2.legend()

plt.tight_layout()
plt.savefig('data/output/comparison.png')

def metric_eval (y_test, predictions) -> pd.Series:
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    return pd.Series({
        'rmse': rmse,
        'mae': mae,
        'mape %': mape*100
    })

xgbm = metric_eval(load_aligned,xgb['predicted_load'])
bm = metric_eval(load_aligned,baseline['predicted_load'])
metrics = pd.DataFrame([xgbm, bm], index=['XGBoost', 'Baseline']).round(2)
metrics.index.name = "Model"
metrics.columns = ["RMSE", "MAE", "MAPE (%)"]
metrics_t = metrics.T
fig, ax = plt.subplots(figsize=(6, 1.8))
ax.axis('off')

tbl = ax.table(
    cellText=metrics_t.astype(str).values.tolist(),
    rowLabels=metrics_t.index.tolist(),
    colLabels=metrics_t.columns.tolist(),
    loc='center',
    cellLoc='center'
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.3)

plt.tight_layout()
plt.savefig('data/output/metrics.png')

