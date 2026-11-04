"""Reproduce Report 2. Python 3; see requirements.txt. No identifiers are predictors."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import json,time,hashlib,platform,urllib.request,zipfile
import numpy as np,pandas as pd,scipy,sklearn
try:
 import lightgbm
 from lightgbm import LGBMClassifier
except ImportError:
 lightgbm=None
from sklearn.model_selection import GroupShuffleSplit,StratifiedShuffleSplit,StratifiedGroupKFold,StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,precision_score,recall_score,f1_score,brier_score_loss
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
SEEDS=[int(os.environ['RUN_SEED'])] if 'RUN_SEED' in os.environ else [11,23,42,67,101]
NUM=['time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient','number_diagnoses']
EXCLUDE=[11,13,14,19,20,21]
URL='https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip'
def diagnosis(v):
 if v=='?':return 'Unknown'
 if str(v).startswith(('E','V')):return 'Other'
 try:n=float(v)
 except ValueError:return 'Unknown'
 if 250<=n<251:return 'Diabetes'
 for lo,hi,extra,name in [(390,460,785,'Circulatory'),(460,520,786,'Respiratory'),(520,580,787,'Digestive'),(580,630,788,'Genitourinary'),(800,1000,-1,'Injury'),(710,740,-1,'Musculoskeletal'),(140,240,-1,'Neoplasm')]:
  if lo<=n<hi or int(n)==extra:return name
 return 'Other'
def load():
 p=ROOT/'data/diabetic_data.csv'
 if not p.exists():
  p.parent.mkdir(exist_ok=True);z=p.parent/'uci.zip';urllib.request.urlretrieve(URL,z)
  with zipfile.ZipFile(z) as zz:zz.extractall(p.parent)
 raw=pd.read_csv(p,keep_default_na=False)
 assert set(raw.readmitted)=={'<30','>30','NO'}
 df=raw.loc[~raw.discharge_disposition_id.isin(EXCLUDE)].copy()
 counts=df.groupby('patient_nbr').size()
 audit={'raw_rows':len(raw),'raw_patients':raw.patient_nbr.nunique(),'raw_columns':len(raw.columns),'raw_positive':int((raw.readmitted=='<30').sum()),'rows':len(df),'patients':df.patient_nbr.nunique(),'positive':int((df.readmitted=='<30').sum()),'excluded':len(raw)-len(df),'exclude_codes':EXCLUDE,'exact_duplicates':int(raw.duplicated().sum()),'duplicate_encounter_ids':int(raw.encounter_id.duplicated().sum()),'duplicate_nonidentifier_rows':int(raw.drop(columns=['encounter_id','patient_nbr']).duplicated().sum()),'repeat_patients':int((counts>1).sum()),'repeat_patient_encounters':int(counts[counts>1].sum()),'encounters_quantiles':counts.quantile([.5,.9,.95,.99,1]).to_dict(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 pd.DataFrame({'encounters':counts}).value_counts().sort_index().rename('patients').to_csv(OUT/'encounters_per_patient.csv')
 pd.DataFrame([{'variable':c,'unknown_count':int((df[c]=='?').sum()),'not_performed_count':int((df[c]=='None').sum()) if c in ['A1Cresult','max_glu_serum'] else 0,'unknown_pct':100*(df[c]=='?').mean()} for c in df]).to_csv(OUT/'missingness.csv',index=False)
 df.describe(include='all').T.to_csv(OUT/'variable_summary.csv')
 pd.DataFrame([{'variable':c,'value':str(v),'count':int(n)} for c in df if c not in ['patient_nbr','encounter_id'] for v,n in df[c].value_counts().items()]).to_csv(OUT/'category_counts.csv',index=False)
 pd.Series(audit).to_json(OUT/'audit.json',indent=2)
 X=df.drop(columns=['encounter_id','patient_nbr','readmitted','weight']).copy()
 for c in ['diag_1','diag_2','diag_3']:X[c]=X[c].map(diagnosis)
 for c in X:
  if c not in NUM:X[c]=X[c].astype(str).replace({'?':'Unknown','None':'Not performed'})
 assert not X.isna().any().any()
 return df,X,(df.readmitted=='<30').astype(int).to_numpy()
def preprocessor(X):
 return ColumnTransformer([('numeric',StandardScaler(),NUM),('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=True),[c for c in X if c not in NUM])],sparse_threshold=1)
def model(name,config,seed):
 if name in ['L2','L1','L2_unweighted']:
  return LogisticRegression(C=config,penalty='l1' if name=='L1' else 'l2',solver='liblinear',class_weight=None if name=='L2_unweighted' else 'balanced',max_iter=1000,random_state=seed)
 if name=='RF':return RandomForestClassifier(n_estimators=100,max_depth=12,min_samples_leaf=config,max_features='sqrt',class_weight='balanced',n_jobs=2,random_state=seed)
 return LGBMClassifier(n_estimators=150,learning_rate=.05,num_leaves=config,max_depth=-1,min_child_samples=50,class_weight='balanced',verbosity=-1,n_jobs=2,random_state=seed,deterministic=True,force_col_wise=True)
CONFIG={'L2':[.1,1.0],'L1':[.01,.1],'RF':[5,20],'LGBM':[15,31],'L2_unweighted':[.1,1.0]}
def metrics(y,p,t):
 return {'ap':average_precision_score(y,p),'roc_auc':roc_auc_score(y,p),'precision':precision_score(y,p>=t,zero_division=0),'recall':recall_score(y,p>=t,zero_division=0),'f1':f1_score(y,p>=t,zero_division=0),'brier':brier_score_loss(y,p)}
def run():
 df,X,y=load();groups=df.patient_nbr.to_numpy();rows=[];splits=[];tuning=[]
 for seed in SEEDS:
  for arm in ['patient','encounter']:
   target=OUT/f'predictions_{arm}_{seed}.csv.gz'
   if target.exists():
    rows.extend(json.loads((OUT/f'metrics_{arm}_{seed}.json').read_text()));continue
   start=time.time()
   splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed) if arm=='patient' else StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=seed)
   tr,te=next(splitter.split(X,y,groups));gtr=groups[tr]
   overlap=np.isin(groups[te],gtr)
   if arm=='patient':assert not overlap.any()
   splitrow={'arm':arm,'seed':seed,'train_n':len(tr),'test_n':len(te),'train_patients':len(np.unique(gtr)),'test_patients':len(np.unique(groups[te])),'train_prevalence':float(y[tr].mean()),'test_prevalence':float(y[te].mean()),'overlap_encounter_pct':float(100*overlap.mean()),'overlap_patients':len(set(gtr)&set(groups[te]))}
   (OUT/f'split_{arm}_{seed}.json').write_text(json.dumps(splitrow,indent=2))
   pd.DataFrame({'encounter_id':df.encounter_id,'partition':np.where(np.isin(np.arange(len(X)),te),'test','development')}).to_csv(OUT/f'partition_{arm}_{seed}.csv.gz',index=False)
   cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed) if arm=='patient' else StratifiedKFold(5,shuffle=True,random_state=seed)
   oof={(n,c):np.zeros(len(tr)) for n,cs in CONFIG.items() for c in cs}
   scores={key:[] for key in oof}
   for fold,(a,b) in enumerate(cv.split(X.iloc[tr],y[tr],gtr)):
    if arm=='patient':assert not set(gtr[a])&set(gtr[b])
    pp=preprocessor(X);A=pp.fit_transform(X.iloc[tr[a]]);B=pp.transform(X.iloc[tr[b]])
    for (n,c),pred in oof.items():
     print(f'fit {arm} {seed} fold={fold+1} {n} C={c}',flush=True);m=model(n,c,seed);aa=A.toarray().astype('float32') if n=='RF' else A;bb=B.toarray().astype('float32') if n=='RF' else B;m.fit(aa,y[tr[a]]);p=m.predict_proba(bb)[:,1];pred[b]=p;scores[n,c].append(float(average_precision_score(y[tr[b]],p)))
    print(f'{arm} seed={seed} fold={fold+1}/5 elapsed={time.time()-start:.1f}s',flush=True)
   pp=preprocessor(X);A=pp.fit_transform(X.iloc[tr]);B=pp.transform(X.iloc[te]);preds=pd.DataFrame({'encounter_id':df.encounter_id.iloc[te].to_numpy(),'patient_nbr':groups[te],'y':y[te],'seen_in_training':overlap})
   runrows=[]
   for n,cs in CONFIG.items():
    best=max(cs,key=lambda c:np.mean(scores[n,c]));pval=oof[n,best];prec,rec,threshold=precision_recall_curve(y[tr],pval);f=2*prec[:-1]*rec[:-1]/np.maximum(prec[:-1]+rec[:-1],1e-15);t=float(threshold[np.argmax(f)])
    m=model(n,best,seed);aa=A.toarray().astype('float32') if n=='RF' else A;bb=B.toarray().astype('float32') if n=='RF' else B;m.fit(aa,y[tr]);p=m.predict_proba(bb)[:,1];preds[n]=p
    row={'arm':arm,'seed':seed,'model':n,'config':best,'threshold':t,'cv_ap':float(np.mean(scores[n,best])),'features':A.shape[1],**metrics(y[te],p,t)};runrows.append(row)
    for c in cs:tuning.append({'arm':arm,'seed':seed,'model':n,'candidate':c,'fold_ap':scores[n,c]})
   baseline=np.full(len(te),y[tr].mean());preds['constant']=baseline;runrows.append({'arm':arm,'seed':seed,'model':'constant','config':0,'threshold':.5,'cv_ap':float(y[tr].mean()),'features':0,**metrics(y[te],baseline,.5)})
   preds.to_csv(target,index=False,compression='gzip');(OUT/f'metrics_{arm}_{seed}.json').write_text(json.dumps(runrows,indent=2));rows.extend(runrows)
   (OUT/f'tuning_{arm}_{seed}.json').write_text(json.dumps([v for v in tuning if v['arm']==arm and v['seed']==seed],indent=2))
   print('COMPLETE',arm,seed,'seconds',round(time.time()-start),flush=True)
 pd.DataFrame(rows).to_csv(OUT/('metrics_'+str(SEEDS[0])+'.csv' if 'RUN_SEED' in os.environ else 'metrics.csv'),index=False)
 pd.DataFrame([json.loads(p.read_text()) for p in sorted(OUT.glob('split_*.json'))]).to_csv(OUT/'splits.csv',index=False)
 env={'python':platform.python_version(),'platform':platform.platform(),'cpu_count':os.cpu_count(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__ if lightgbm else 'not installed in analysis runtime'}
 if not (OUT/'environment.json').exists():(OUT/'environment.json').write_text(json.dumps(env,indent=2))
if __name__=='__main__':run()
