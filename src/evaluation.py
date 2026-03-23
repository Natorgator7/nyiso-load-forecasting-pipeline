from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("evaluation")


#import model outputs
full = pd.read_csv("data/processed/featurized_hourly_load_temp.csv", usecols=["date", "load"])
full["date"] = pd.to_datetime(full["date"], utc=True)
full = full.set_index("date")

baseline = pd.read_csv('data/processed/baseline_predictions.csv', usecols =
                    ['feature_date', 'target_date', 'predicted_load', 'naive_pred', 'actual_load'])
baseline['target_date'] = pd.to_datetime(baseline['target_date'])

xgb = pd.read_csv('data/processed/XGBoost_predictions.csv', usecols=
                    ['target_date', 'predicted_load', 'feature_date', 'actual_load'])
xgb['target_date'] = pd.to_datetime(xgb['target_date'])


baseline = baseline.dropna().set_index("target_date")
xgb = xgb.dropna().set_index("target_date")

#graphing
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 8), sharex=False)

#subplot 1: actual vs baseline
full['load'].plot(ax=ax1, label='Actual Load', alpha=0.4)
baseline['predicted_load'].plot(ax=ax1, label='Baseline Pred', alpha=0.6)
ax1.set_title('Actual vs Baseline Prediction')
ax1.set_xlabel('Date')
ax1.set_ylabel('Load (MW)')
ax1.legend()

#subplot 2: actual vs XGBoost
full['load'].plot(ax=ax2, label='Actual Load', alpha=0.4)
xgb['predicted_load'].plot(ax=ax2, label='XGBoost Pred', alpha=0.6, color = 'green')
ax2.set_title('Actual vs XGBoost Prediction')
ax2.set_xlabel('Date')
ax2.set_ylabel('Load (MW)')
ax2.legend()

#subplot 3: actual vs. naive
full['load'].plot(ax=ax3, label='Actual Load', alpha=0.4)
baseline['naive_pred'].plot(ax=ax3, label='Naive Pred', alpha=0.6, color = 'purple')
ax3.set_title('Actual vs Naive Prediction')
ax3.set_xlabel('Date')
ax3.set_ylabel('Load (MW)')
ax3.legend()

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

xgbm = metric_eval(xgb['actual_load'],xgb['predicted_load'])
bm = metric_eval(baseline['actual_load'],baseline['predicted_load'])
nm = metric_eval(baseline['actual_load'],baseline['naive_pred'])
metrics = pd.DataFrame([xgbm, bm, nm], index=['XGBoost', 'Baseline', 'Naive']).round(2)
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

