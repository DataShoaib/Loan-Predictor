import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from uttils.logger import getlogger
import os


logger=getlogger('data_ingestion')

def load_data(path:str) -> pd.DataFrame:
    try:
        logger.info(f"Attempting to load data from path: {path}")
        data=pd.read_csv(path)
        logger.info(f"Successfully loaded data with shape: {data.shape}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found at path: {path}")
    except Exception as e:
        logger.error(f'an error occurred while loading data: {e}')   

def clean_data(data:pd.DataFrame)-> pd.DataFrame:
    try:
        logger.info('starting data cleaning process')
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        data.columns = data.columns.str.strip()
        # we can replace the negative value with 0 as it is not possible to have negative assets value 
        data['residential_assets_value']= data['residential_assets_value'].apply(lambda x:0 if x<0 else x)
        logger.info('data cleaning process completed successfully')
        return data
    except Exception as e:
        logger.error(f'an error occurred while cleaning data: {e}')

def save_data(data:pd.DataFrame,path:str):
    try:
        logger.info(f'attempting to save cleaned data to path: {path}')
        path=os.path.join(path,'cleaned_data.csv')
        data.to_csv(path,index=False)
        logger.info(f'successfully saved cleaned data to path: {path}')
    except Exception as e:
        logger.error(f'an error occurred while saving data: {e}')

def main():
    data=load_data('data/raw/loan_approval_prediction.csv')
    if data is not None:
        data=clean_data(data)
        save_data(data,'data/processed')