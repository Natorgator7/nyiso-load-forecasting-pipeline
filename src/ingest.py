import logging
import pandas as pd
import numpy as np
from datetime import datetime
import glob



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

#ingest all load datasets
load_files = glob.glob('data/Load/*.csv')
dfs = [ingest_csv(f, ['RTD End Time Stamp', 'RTD Actual Load']) for f in load_files]
load = pd.concat(dfs, ignore_index=True)

#ingest all temp datasets
temp = ingest_csv('data/Temp/Temp_March_23_2020-2026.csv', ['valid', 'tmpf'])

#clean for readability
load.columns = ['date', 'load']
temp.columns = ['date', 'temperature']
temp['date'] = pd.to_datetime(temp['date'], utc = True)
load['date'] = pd.to_datetime(load['date'], utc = True)
temp['temperature'] = pd.to_numeric(temp['temperature'], errors='coerce')

load = load.set_index('date')
temp = temp.set_index('date')

#matching rows
load_hourly = load.resample("h").mean()
#load_hourly = load_hourly.iloc[:-1]
temp_hourly = temp.resample("h").mean()

df = load_hourly.join(temp_hourly, how="inner")

#log row matching
logger.info(f"merged_dataset rows={len(df)}")
missing = df.isna().sum()
logger.info(f"missing_values {missing.to_dict()}")

df.to_csv("data/processed/hourly_load_temp.csv")
logger.info("saved_processed_dataset")
# %%
