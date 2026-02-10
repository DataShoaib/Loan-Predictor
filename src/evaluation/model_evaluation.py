from sklearn.metrics import classification_report,confusion_matrix,precision_score
from sklearn.metrics import recall_score,f1_score,accuracy_score,precision_recall_curve,roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from uttils.logger import get_logger
import json

logger=get_logger('model_evaluation')

def load_test_data(x_test_path:str,y_test_path:str)->tuple:
    try:
        x_test=pd.read_csv(x_test_path)
        y_test=pd.read_csv(y_test_path)
        x_test['loan_to_income'] = x_test['loan_amount'] / x_test['income_annum']
        x_test = x_test.drop(columns=['loan_amount', 'income_annum'])
        return x_test,y_test
    except Exception as e:
        logger.error(f'an error accured during loading test data: {e}')
        raise Exception('failed to load test data')
    
def load_model(model_path:str):
    try:
        model=pd.read_pickle(model_path)
        return model
    except Exception as e:
        logger.error(f"an error accured during the loading model:{e}")   
      

def evaluate_model(model:Pipeline,x_test:pd.DataFrame,y_test:pd.DataFrame)->dict:     
    try:
        logger.info('starting model evaluation')
        y_prob=model.predict_proba(x_test)[:,1]
        # 0.53 is chosen as the threshold based on precision-recall curve anakysis during the experimentation phase.
        y_pred=(y_prob>0.53).astype(int)
        acc=accuracy_score(y_test,y_pred)
        prec=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        f1_scr=f1_score(y_test,y_pred)
        metrices_info={"accuracy_score":acc,"precision_score":prec,"recall_score":recall,"f1_score":f1_scr}
        return metrices_info
        logger.info(f"Model evaluation results: Accuracy={acc}, Precision={prec}, Recall={recall}, F1 Score={f1_scr}")
    except Exception as e:  
        logger.error(f'an error accured while model evaluation: {e}')  


def saving_metric_info(metric_info:dict,save_path:str)->None:
    try:
        logger.info(f'saving metric information to path:{save_path}') 
        with open(save_path,'w') as f:
            json.dump(metric_info,f,indent=4)   
        logger.info(f'file save successfully at path:{save_path}')     
    except Exception as e:
        logger.error(f'an error accured while saving the metric_info: {e}')   

def plot_confusion_matrix(model:Pipeline,x_test:pd.DataFrame,y_test:pd.DataFrame,save_path:str)->None:
    try:
        logger.info('starting confusion matrix plotting')        
        y_prob=model.predict_proba(x_test)[:,1]
        y_pred=(y_prob>0.53).astype(int)
        cm=confusion_matrix(y_test,y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm,annot=True,fmt='d')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(save_path)
        logger.info(f'Confusion matrix saved to {save_path}')
    except Exception as e:
        logger.error(f'an error accured while plotting confusion matrix: {e}')

def main():
    x_test,y_test=load_test_data('data/test_data/x_test.csv','data/test_data/y_test.csv')    
    model=load_model("models/model_pipe.pkl")  
    metric_info=evaluate_model(model,x_test,y_test)  
    saving_metric_info(metric_info,'reports/metric_info.json')
    plot_confusion_matrix(model,x_test,y_test,'reports/figures/cm.png')

if __name__=="__main__":
    main()    