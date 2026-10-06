import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app,Booking,bundle
from src.features import features,FEATURES,FORBIDDEN,split_masks
client=TestClient(app)
def test_allowlist_blocks_target_and_late_fields():
    row=Booking().model_dump(); row.update(is_canceled=1,reservation_status='Canceled',adr=900)
    x=features(pd.DataFrame([row]));assert list(x)==FEATURES;assert not set(x)&FORBIDDEN
    assert x.total_nights.iloc[0]==4 and x.total_guests.iloc[0]==2

def test_split_excludes_future_labels():
    dates=pd.Series(pd.to_datetime(['2016-09-20','2016-10-20','2017-01-20','2017-03-20']))
    raw=pd.DataFrame({'reservation_status_date':['2016-10-02','2016-10-22','2017-01-22','2017-03-22']})
    masks=split_masks(raw,dates)
    assert not masks['train'].iloc[0]
    assert sum(m.sum() for m in masks.values())==3
    assert all(sum(int(m.iloc[i]) for m in masks.values())<=1 for i in range(4))

def test_api_rejects_invalid_and_leaky_input():
    for data in [{'lead_time':-1},{'arrival_month':13},{'adults':0,'children':0,'babies':0},{'is_canceled':1},{'hotel':'unknown'}]:
        assert client.post('/predict',json=data).status_code==422

def test_api_prediction_contract():
    class Fake:
        def predict_proba(self,x):return np.array([[.3,.7]])
    app.dependency_overrides.clear()
    import app.main as main
    old=main.bundle;main.bundle=lambda:{'model':Fake(),'threshold':.4,'selected_model':'test'}
    try:
        r=client.post('/predict',json={});assert r.status_code==200
        assert r.json()['review_recommended'] and r.json()['cancellation_probability']==.7
    finally:main.bundle=old

def test_live_model_and_unknown_category():
    b=bundle();x=features(pd.DataFrame([Booking().model_dump()]));x['meal']='new_meal'
    p=b['model'].predict_proba(x)[0,1];assert np.isfinite(p) and 0<=p<=1
    assert client.post('/predict',json={}).status_code==200
