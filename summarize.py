from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.calibration import calibration_curve
from run_experiments import ROOT,OUT,load,NUM
F=ROOT/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
def audit_figures():
 df,_,_=load();c=df.groupby('patient_nbr').size();groups=[(c==1).sum(),(c==2).sum(),c.between(3,5).sum(),(c>=6).sum()]
 fig,ax=plt.subplots(figsize=(3.35,2.0));ax.bar(['1','2','3–5','6+'],groups,color='#28688a');ax.set(xlabel='Encounters per patient',ylabel='Number of patients');ax.ticklabel_format(axis='y',style='plain',useOffset=False)
 for i,v in enumerate(groups):ax.text(i,v+1000,f'{v:,}',ha='center',fontsize=8)
 ax.set_ylim(0,max(groups)*1.22);fig.savefig(F/'patient_counts.pdf');plt.close(fig)
 fig,ax=plt.subplots(figsize=(7.1,2.45));fig.subplots_adjust(left=.01,right=.99,bottom=.1,top=.98);ax.set(xlim=(-.1,10),ylim=(0,3));ax.axis('off')
 def box(x,y,w,h,s,color='#e7f0f5'):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',facecolor=color,edgecolor='#35566a'));ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=9)
 def arr(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#35566a','lw':1.2})
 box(.05,1.08,1.65,.85,'Fixed cohort\n+ code mapping');box(2.02,1.08,1.5,.85,'Outer split\nby patient\nor encounter');arr((1.76,1.5),(1.96,1.5))
 box(3.9,1.65,2.35,1.1,'Development: 5-fold CV\nFit encoding, scaling,\nweights and model in\ntraining folds only');arr((3.58,1.6),(3.84,2.15))
 box(6.65,1.65,2.85,1.1,'Validation selects settings\nand decision threshold\nRefit development data');arr((6.31,2.2),(6.59,2.2))
 box(3.9,.15,2.35,.68,'Untouched test data',color='#fff1dc');arr((3.58,1.4),(3.84,.49))
 box(6.65,.15,2.85,.68,'Apply frozen pipeline\nScore without refitting',color='#fff1dc');arr((6.31,.49),(6.59,.49));arr((8.08,1.59),(8.08,.89))
 ax.text(0,-.12,'Repeat for 5 predefined seeds. L1 selection is training-only; SMOTENC remains planned.',fontsize=8)
 fig.savefig(F/'workflow.pdf');plt.close(fig)
def results():
 m=pd.read_csv(OUT/'metrics.csv');s=m.groupby(['arm','model'])[['ap','roc_auc','precision','recall','f1','brier']].agg(['mean','std']);s.to_csv(OUT/'summary.csv')
 names=['L2','L1','RF','LGBM'];fig,ax=plt.subplots(figsize=(3.35,2.25))
 for j,(arm,col) in enumerate([('patient','#28688a'),('encounter','#c56634')]):
  for i,n in enumerate(names):
   vals=m[(m.arm==arm)&(m.model==n)].ap.to_numpy();x=i+(j-.5)*.22
   ax.scatter(x+np.linspace(-.035,.035,len(vals)),vals,color=col,s=13,alpha=.7);ax.errorbar(x,vals.mean(),yerr=vals.std(ddof=1),fmt='_',color=col,capsize=3,label=arm if i==0 else None)
 ax.set(xticks=range(4),xticklabels=['L2 LR','L1 LR','RF','LightGBM'],ylabel='Average precision');ax.legend(fontsize=8);fig.savefig(F/'comparison.pdf');plt.close(fig)
 p=pd.read_csv(OUT/'predictions_patient_42.csv.gz');fig,ax=plt.subplots(figsize=(3.35,2.1))
 for n in ['L2','L2_unweighted','LGBM']:
  true,pred=calibration_curve(p.y,p[n],n_bins=8,strategy='quantile');ax.plot(pred,true,'o-',ms=3,label={'L2':'Weighted L2 logistic','L2_unweighted':'Unweighted L2 logistic','LGBM':'LightGBM'}[n])
 ax.plot([0,1],[0,1],'--',color='gray',lw=.8);ax.set(xlabel='Mean predicted probability',ylabel='Observed positive fraction',xlim=(0,.8),ylim=(0,.8));ax.legend(fontsize=7);fig.savefig(F/'calibration.pdf');plt.close(fig)
 # Prespecified seed 42, paired cluster bootstrap, same patients across models.
 rng=np.random.default_rng(2026);ids,inv=np.unique(p.patient_nbr,return_inverse=True);records=[]
 for b in range(1000):
  multiplicity=np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids));w=multiplicity[inv];base=average_precision_score(p.y,p.L2,sample_weight=w)
  records.append({n:average_precision_score(p.y,p[n],sample_weight=w)-base for n in ['L1','RF','LGBM','L2_unweighted']})
 ci=[]
 for n in records[0]:
  v=np.array([r[n] for r in records]);ci.append({'model':n,'ap_difference':average_precision_score(p.y,p[n])-average_precision_score(p.y,p.L2),'lower':np.quantile(v,.025),'upper':np.quantile(v,.975),'replicates':1000,'seed':42})
 pd.DataFrame(ci).to_csv(OUT/'bootstrap.csv',index=False)
 seen=[]
 for file in sorted(OUT.glob('predictions_encounter_*.csv.gz')):
  p=pd.read_csv(file)
  for status,g in p.groupby('seen_in_training'):
   for n in names:seen.append({'seed':int(file.name.split('_')[-1].split('.')[0]),'seen':bool(status),'n':len(g),'prevalence':g.y.mean(),'model':n,'ap':average_precision_score(g.y,g[n]),'roc_auc':roc_auc_score(g.y,g[n])})
 pd.DataFrame(seen).to_csv(OUT/'seen_unseen.csv',index=False)
 print(s.to_string());print(pd.DataFrame(ci).to_string(index=False))
if __name__=='__main__':
 audit_figures()
 if (OUT/'metrics.csv').exists():results()
