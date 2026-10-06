import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse, hashlib, json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,brier_score_loss,confusion_matrix,roc_curve,precision_recall_curve)
from src.features import *
ROOT=Path(__file__).resolve().parents[1]

def metrics(y,p,t=.5):
    h=p>=t
    return { 'accuracy':float(accuracy_score(y,h)), 'precision':float(precision_score(y,h,zero_division=0)), 'recall':float(recall_score(y,h,zero_division=0)), 'f1':float(f1_score(y,h,zero_division=0)), 'roc_auc':float(roc_auc_score(y,p)), 'average_precision':float(average_precision_score(y,p)), 'brier':float(brier_score_loss(y,p)), 'confusion_matrix':confusion_matrix(y,h,labels=[0,1]).tolist() }

def make_pipeline(model):
    prep=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUMERIC),('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CATEGORICAL)])
    return Pipeline([('preprocess',prep),('model',model)])

def main(path):
    raw,dates,audit=load_data(path); X=features(raw); y=raw.is_canceled
    masks=split_masks(raw,dates)
    tr,va,ca,te=[masks[k] for k in ['train','validation','calibration','test']]
    report={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'audit':audit,'prediction_scope':'Retrospective arrival-cohort study; snapshot data does not validate booking-time deployment.',
        'features':FEATURES,'excluded_features':sorted(FORBIDDEN),'selection_metric':'Validation average precision',
        'splits':{k:{'n':int(m.sum()),'cancellation_rate':float(y[m].mean()),'arrival_min':str(dates[m].min().date()),'arrival_max':str(dates[m].max().date())} for k,m in masks.items()},
        'outcomes_purged_at_boundaries':int(len(raw)-sum(m.sum() for m in masks.values())), 'validation_models':{}}
    fitted={}
    candidates={'dummy':DummyClassifier(strategy='prior'),'logistic':LogisticRegression(max_iter=1500,C=1),'random_forest':RandomForestClassifier(n_estimators=120,max_depth=14,min_samples_leaf=20,n_jobs=2,random_state=42),'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,l2_regularization=5,learning_rate=.06,early_stopping=False,random_state=42)}
    for name,model in candidates.items():
        start=time.perf_counter(); pipe=make_pipeline(model);pipe.fit(X[tr],y[tr]);fitted[name]=pipe
        report['validation_models'][name]={**metrics(y[va],pipe.predict_proba(X[va])[:,1]),'fit_seconds':round(time.perf_counter()-start,2)}
        print(name,report['validation_models'][name],flush=True)
    winner=max((k for k in candidates if k!='dummy'),key=lambda k:report['validation_models'][k]['average_precision'])
    # Freeze training model; no refitting on test or calibration features.
    calibrated=CalibratedClassifierCV(FrozenEstimator(fitted[winner]),method='sigmoid')
    calibrated.fit(X[ca],y[ca]); cp=calibrated.predict_proba(X[ca])[:,1]
    grid=np.arange(.05,.951,.01)
    # Illustrative missed-cancellation cost 5x false alert, not currency/savings.
    costs=[5*np.sum((cp<t)&(y[ca].to_numpy()==1))+np.sum((cp>=t)&(y[ca].to_numpy()==0)) for t in grid]
    threshold=float(grid[int(np.argmin(costs))])
    report.update(selected_model=winner,threshold=threshold,cost_assumption={'false_negative':5,'false_positive':1,'unit':'illustrative penalty units'},threshold_selection='Calibration cohort; same cohort used for sigmoid fit. Final test remains untouched.')
    tp=calibrated.predict_proba(X[te])[:,1]; ty=y[te]
    report['test_calibrated']=metrics(ty,tp,threshold)
    report['test_at_0_5']=metrics(ty,tp,.5)
    report['test_uncalibrated']=metrics(ty,fitted[winner].predict_proba(X[te])[:,1],.5)
    report['test_dummy']=metrics(ty,fitted['dummy'].predict_proba(X[te])[:,1],.5)
    report['test_by_hotel']={h:metrics(ty[raw.loc[te,'hotel']==h],tp[(raw.loc[te,'hotel']==h).to_numpy()],threshold) for h in raw.hotel.unique()}
    # Day-cluster bootstrap quantifies temporal dependence more conservatively than row bootstrap.
    rng=np.random.default_rng(42); groups=[np.flatnonzero(dates[te].to_numpy()==d) for d in dates[te].unique()]; estimates=[]
    for _ in range(200):
        ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]); yy=ty.to_numpy()[ix];pp=tp[ix]
        estimates.append([accuracy_score(yy,pp>=threshold),roc_auc_score(yy,pp),average_precision_score(yy,pp)])
    report['test_day_cluster_bootstrap_95']={k:np.quantile(np.array(estimates)[:,i],[.025,.975]).tolist() for i,k in enumerate(['accuracy','roc_auc','average_precision'])}
    # Importance is validation-only, before final-test interpretation.
    sample=X[va].sample(min(2500,int(va.sum())),random_state=42)
    imp=permutation_importance(fitted[winner],sample,y.loc[sample.index],scoring='average_precision',n_repeats=3,random_state=42,n_jobs=2)
    importance=pd.DataFrame({'feature':FEATURES,'mean_ap_drop':imp.importances_mean,'std':imp.importances_std}).sort_values('mean_ap_drop',ascending=False)
    importance.to_csv(ROOT/'reports/feature_importance.csv',index=False)
    report['importance_note']='Validation permutation importance on engineered columns; correlated raw/derived features can dilute importance; associations are not causal.'
    fig,axs=plt.subplots(2,2,figsize=(12,9)); fig.suptitle('Hotel cancellation — untouched March–August 2017 cohort')
    fpr,tpr,_=roc_curve(ty,tp);axs[0,0].plot(fpr,tpr,label=f"ROC AUC {report['test_calibrated']['roc_auc']:.3f}");axs[0,0].plot([0,1],[0,1],'--',color='gray');axs[0,0].set(xlabel='False positive rate',ylabel='True positive rate');axs[0,0].legend()
    pre,rec,_=precision_recall_curve(ty,tp);axs[0,1].plot(rec,pre);axs[0,1].axhline(ty.mean(),ls='--',color='gray');axs[0,1].set(xlabel='Recall',ylabel='Precision',title='Precision–recall; dashed = prevalence')
    frac,means=calibration_curve(ty,tp,n_bins=10,strategy='quantile');axs[1,0].plot(means,frac,'o-');axs[1,0].plot([0,1],[0,1],'--',color='gray');axs[1,0].set(xlabel='Predicted probability',ylabel='Observed rate',title='Calibration on future cohort')
    top=importance.head(8).iloc[::-1];axs[1,1].barh(top.feature,top.mean_ap_drop);axs[1,1].set(title='Validation permutation importance',xlabel='Average precision decrease');fig.tight_layout();fig.savefig(ROOT/'reports/evaluation.png',dpi=160);plt.close(fig)
    preds=pd.DataFrame({'arrival_date':dates[te].dt.strftime('%Y-%m-%d'),'hotel':raw.loc[te,'hotel'],'label':ty,'probability':tp,'flagged':tp>=threshold})
    preds.to_csv(ROOT/'reports/test_predictions.csv',index=False)
    report['data_summary']=raw.groupby('hotel').is_canceled.agg(['count','mean']).to_dict('index')
    (ROOT/'reports/metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    joblib.dump({'model':calibrated,'threshold':threshold,'features':FEATURES,'selected_model':winner},ROOT/'models/cancellation.joblib',compress=3)
    print('SELECTED',winner,'TEST',report['test_calibrated'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data/hotels.csv');args=p.parse_args();main(args.data)
