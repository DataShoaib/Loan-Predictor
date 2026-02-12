from fastapi import FastAPI
from app.config import settings
from app.model import load_model,predict
from app.schema import Loanrequest,Loanresponse
from sklearn.pipeline import Pipeline

model = load_model(settings.MODEL_PATH)
   
app=FastAPI(title=settings.APP_NAME,version=settings.VERSION,)

@app.get('/health')
def health_check():
    return {"status": "Loan Predictor API running 🚀"}


@app.post("/predict",response_model=Loanresponse)
def prediction(data:Loanrequest):
    result=predict(model,data)
    return result 

    
