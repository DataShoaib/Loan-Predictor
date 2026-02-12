from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel,Field
from enum import Enum
from typing import Literal

class Education(str,Enum):
    graduate="Graduate"
    not_graduate='not_Graduate'


class   Loanrequest(BaseModel):
    no_of_dependents: int =Field(...,title='dependents to appli',ge=0,description="number of dependents to the applilcant who ai'nt earn",examples=[1,2,3,4])
    education:Education=Field(...,title='applicant education',description='education of the applicant',examples=['Graduate',"not_graduate"])
    self_employed:Literal["Yes","No"]=Field(..., title='self_employed or not',description="applicant is self employed or not",examples=['yes','No'])
    income_annum:int=Field(...,gt=0,title='income_per_year',description='total income of applicant per year')
    loan_amount :int=Field(...,gt=0,title='loan amount',description='loan amount of applicant')
    loan_term:int=Field(...,ge=0,le=20,title='tenure for the loan',description='for how much year loan taken for',examples=[2,3,4,5])
    cibil_score: int =Field(...,ge=300,le=900,title='cibil-score' ,description='past performance of the applicant with bank',examples=[450,390,700])
    residential_assets_value:int=Field(...,ge=0,title='personal property',description='personal property of the applicant')
    commercial_assets_value:int=Field(...,ge=0,title='commercial property',description='commercial(bussness) property of the applicant')
    luxury_assets_value:int=Field(...,ge=0,title='luxury items',description='luxury items value of the applicant ')
    bank_asset_value:int=Field(...,ge=0,title='bank assets value',description='bank asset value of the applicant')

class Loanresponse(BaseModel):
    loan_approval_status:str 
    probablity:float

