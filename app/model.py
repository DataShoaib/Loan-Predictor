import pandas as pd
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from uttils.logger import get_logger


logger=get_logger('fast_api')

def load_model(model_path:str)->Pipeline:
    try:
        logger.info(f'try to loading the model:{model_path}')
        with open(model_path,'rb') as f:
            model=pickle.load(f)
        logger.info(f'model loaded successfully from:{model_path}')
        return model
    except FileNotFoundError as e:
        logger.error(f'file not found at {model_path}')
    except Exception as e:
        logger.error(f'an error accured during the loading model:{e}')


def predict(model:Pipeline,data)->dict:

    input_data = {
        "no_of_dependents": data.no_of_dependents,
        "education": data.education,
        "self_employed": data.self_employed,
        "income_annum": data.income_annum,
        "loan_amount": data.loan_amount,
        "loan_term": data.loan_term,
        "cibil_score": data.cibil_score,
        "residential_assets_value": data.residential_assets_value,
        "commercial_assets_value": data.commercial_assets_value,
        "luxury_assets_value": data.luxury_assets_value,
        "bank_asset_value": data.bank_asset_value,
    }
    logger.info('data retrieved successfully and feature eng start')
    data=pd.DataFrame([input_data])
    data['loan_to_income'] = data['loan_amount'] / data['income_annum']
    data = data.drop(columns=['loan_amount', 'income_annum'])
    logger.info(f'feature eng. done successfully')

    y_prob=model.predict_proba(data)[0][1]
    y_pred='approved'if(y_prob> 0.53) else 'not approved'

    logger.info(f'model predction:{y_pred} and probablity:{y_prob}')

    return {
        'loan_approval_status': y_pred,
        'probablity':round(float(y_prob),3)
    }

if __name__=="__main__":
    predict()
     

   
   


