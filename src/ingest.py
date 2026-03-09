
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


#set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ingestion")

def ingest_csv(path: str, usecols) -> pd.DataFrame:
    start = datetime.now()
    
    #read
    df = pd.read_csv(path, usecols = usecols)
    row_count = len(df)
    
    #time
    end = datetime.now()
    duration_sec = (end - start).total_seconds()
    
    #data lengths
    logger.info(
        f"ingestion_completed rows={row_count} time={duration_sec:.2f}s file={path}")

    return df

#read CSVs from NYISO actual load and ASOS temperature
load = ingest_csv('data/Load_March_8_2025-2026.csv', ['RTD End Time Stamp', 'RTD Actual Load'])
temp = ingest_csv('data/Temp_March_8_2025-2026.csv', ['valid', 'tmpf'])


#clean for readability
load.columns = ['date', 'load']
temp.columns = ['date', 'temperature']
temp['date'] = pd.to_datetime(temp['date'])
load['date'] = pd.to_datetime(load['date'])
temp['temperature'] = pd.to_numeric(temp['temperature'], errors='coerce')

load = load.set_index('date')
temp = temp.set_index('date')

#matching rows
load_hourly = load.resample("h").mean()
load_hourly = load_hourly.iloc[:-1]
temp_hourly = temp.resample("h").mean()

df = load_hourly.join(temp_hourly, how="inner")

#plot for visualizing load vs. temperature: df.plot(x='temperature', y='load', kind='scatter')

#log row matching
logger.info(f"merged_dataset rows={len(df)}")
missing = df.isna().sum()
logger.info(f"missing_values {missing.to_dict()}")

df.to_csv("data/processed/hourly_load_temp.csv")
logger.info("saved_processed_dataset")
# %%
