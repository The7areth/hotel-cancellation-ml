from functools import lru_cache
from pathlib import Path
from typing import Literal
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict, model_validator
from src.features import features
ROOT=Path(__file__).resolve().parents[1]
app=FastAPI(title='Hotel Cancellation ML',version='1.0.0')
class Booking(BaseModel):
    model_config=ConfigDict(extra='forbid')
    hotel: Literal['City Hotel','Resort Hotel']='City Hotel'
    lead_time: int=Field(default=60,ge=0,le=800)
    arrival_month: int=Field(default=7,ge=1,le=12)
    stays_in_weekend_nights: int=Field(default=1,ge=0,le=20)
    stays_in_week_nights: int=Field(default=3,ge=0,le=60)
    adults: int=Field(default=2,ge=0,le=55)
    children: int=Field(default=0,ge=0,le=10)
    babies: int=Field(default=0,ge=0,le=10)
    is_repeated_guest: Literal[0,1]=0
    meal: Literal['BB','HB','FB','SC','Undefined']='BB'
    market_segment: Literal['Direct','Corporate','Online TA','Offline TA/TO','Groups','Complementary','Aviation','Undefined']='Online TA'
    distribution_channel: Literal['Direct','Corporate','TA/TO','GDS','Undefined']='TA/TO'
    reserved_room_type: Literal['A','B','C','D','E','F','G','H','L','P']='A'
    customer_type: Literal['Transient','Contract','Transient-Party','Group']='Transient'
    @model_validator(mode='after')
    def validate_totals(self):
        if self.adults+self.children+self.babies<1:raise ValueError('At least one guest is required')
        if self.stays_in_weekend_nights+self.stays_in_week_nights<1:raise ValueError('At least one night is required')
        return self
@lru_cache
def bundle():
    path=ROOT/'models/cancellation.joblib'
    if not path.exists():raise HTTPException(503,'Train the model first: python -m src.download then python -m src.train')
    # Load only this locally trained/trusted artifact. Never deserialize uploads.
    return joblib.load(path)
@app.get('/',include_in_schema=False)
def home():return FileResponse(ROOT/'app/index.html')
@app.get('/health')
def health():return {'status':'ok','model_ready':(ROOT/'models/cancellation.joblib').exists()}
@app.get('/api/metrics')
def report():
    path=ROOT/'reports/metrics.json'
    if not path.exists():raise HTTPException(503,'Train model to generate metrics')
    return json.loads(path.read_text())
@app.post('/predict')
def predict(booking:Booking):
    b=bundle();x=features(pd.DataFrame([booking.model_dump()]));p=float(b['model'].predict_proba(x)[0,1])
    return {'cancellation_probability':p,'review_recommended':p>=b['threshold'],'review_threshold':b['threshold'],
        'model':b['selected_model'],'scope':'Historical two-hotel retrospective model; not a validated live booking-time probability.',
        'action':'Human review only; do not cancel, reprice or overbook automatically.'}
