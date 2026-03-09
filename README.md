Linear regression and XGBoost models for predicting NYISO grid load data 24 hours in advance, using NYISO load and ASOS weather data.

Pipeline:
1. ingest.py
2. features.py
3. train.py
4. evaluation.py

Run:
```bash
python src/main.py

ASOS: https://mesonet.agron.iastate.edu/request/download.phtml?network=NY_ASOS
NYISO: https://www.nyiso.com/load-data