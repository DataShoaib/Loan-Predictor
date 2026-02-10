import pandas as pd 
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from uttils.logger import get_logger
import os
import pickle
 
logger=get_logger('feature_engineering') 

def load_data(data_path:str)->pd.DataFrame:
    try:
        logger.info(f"Loading data from path: {data_path}")
        data=pd.read_csv(data_path)
        logger.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logger.error(f"file not found at path: {data_path}")
    except Exception as e:
        logger.error(f"an error occurred while loading data: {e}")    
        

def splitting_data(data:str)->pd.DataFrame:
    try:
       data['loan_status'] = (
       data['loan_status'].astype(str).str.strip().str.lower() )

       data['loan_status']=data['loan_status'] = data['loan_status'].map({
          'rejected': 0,
          'approved': 1
        })

       logger.info('splitting data into fetures and target variable started')
       x=data.drop('loan_status',axis=1)
       y=data['loan_status']
       x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
       logger.info('data splitting completed successfully')
       return x_train,x_test,y_train,y_test
    except Exception as e:
        logger.error(f"an error occurred while splitting data: {e}")
        raise Exception('data splitting failed')
    

def save_test_data(x_test:pd.DataFrame,y_test:pd.DataFrame,test_data_path:str)-> None:
    try:
        logger.info(f"saving test data to path:{test_data_path}")    
        x_test.to_csv(os.path.join(test_data_path,'x_test.csv'),index=False)
        y_test.to_csv(os.path.join(test_data_path,'y_test.csv'),index=False)
    except Exception as e:
        logger.error(f'an error accured during saving test data: {e}')



def feature_engineering_and_model_training(x_train:pd.DataFrame,y_train:pd.DataFrame)->Pipeline:
    try:
       logger.info('starting feature engineering process')
       # we are using the loan_to_income ratio as a new feature to capture the relationship between loan amount and income, which can be a strong predictor of loan 
       x_train['loan_to_income'] = x_train['loan_amount'] / x_train['income_annum']
       x_train = x_train.drop(columns=['loan_amount', 'income_annum'])
       # log transfromation and satandardization for the right skewed features and ration feature
       log_standard = Pipeline([
       ('log', FunctionTransformer(np.log1p, validate=False)),
      ('scaler', StandardScaler())
      ])

       preprocessor_linear = ColumnTransformer(
        transformers=[
        # right skewed asset values
        ('log_assets', log_standard, [
            'residential_assets_value',
            'commercial_assets_value',
            'bank_asset_value',
            'luxury_assets_value'
        ]),

        # ratio feature (log-scaled)
        ('log_ratio', log_standard, [
            'loan_to_income'
        ]),

        # approximately normal features
        ('normal_scale', StandardScaler(), [
            'cibil_score',
            'loan_term'
        ]),

        ('education_ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), [
            'education'
        ]),

        ('self_employed_bin', OneHotEncoder(drop='if_binary'), [
            'self_employed'
        ])
       ],
       remainder='passthrough')
    #    training
       logger.info('model training staretd')
       model_pipe=Pipeline([('preprocess',preprocessor_linear),('classifier',LogisticRegression(max_iter=200,class_weight='balanced',solver='lbfgs',penalty='l2',C=1.0,random_state=42))])
       model_pipe.fit(x_train,y_train)
       logger.info('model training completed successfully')
       return model_pipe
    except Exception as e:
        logger.error(f'an error occurred during feature engineering: {e}')
        raise Exception('feature engineering or model training failed')
    

def save_model(model:Pipeline,model_path:str)->None:
    try:
        logger.info(f'saving model to path:{model_path}')
        with open(model_path,'wb') as f:
            pickle.dump(model,f)
        logger.info('model saved successfully')
    except Exception as e:
        logger.error(f'an error accurred while saving the model:{e}')
        raise Exception('model saving failed')
    

def main():
    try:
        data=load_data('data/processed/cleaned_data.csv')
        x_train,x_test,y_train,y_test=splitting_data(data)
        save_test_data(x_test,y_test,'data/test_data')
        model=feature_engineering_and_model_training(x_train,y_train)
        save_model(model,'models/model_pipe.pkl')
    except Exception as e:
        logger.error(f'an error occurred in main function: {e}')
        raise Exception('main function execution failed')
    
if __name__=='__main__':
    main()    