"""Integrity checks for the actual exported experiments, not smoke tests."""
import json
import numpy as np,pandas as pd
from run_experiments import ROOT,OUT,load,metrics,SEEDS

def main():
 df,X,y=load();m=pd.read_csv(OUT/'metrics.csv');assert len(m)==60
 assert set(m.seed)=={11,23,42,67,101}
 assert not {'patient_nbr','encounter_id','readmitted','weight'}&set(X.columns)
 for seed in [11,23,42,67,101]:
  for arm in ['patient','encounter']:
   p=pd.read_csv(OUT/f'predictions_{arm}_{seed}.csv.gz');part=pd.read_csv(OUT/f'partition_{arm}_{seed}.csv.gz')
   assert part.encounter_id.nunique()==len(df)==len(part)
   tr=df[df.encounter_id.isin(part.loc[part.partition=='development','encounter_id'])]
   te=df[df.encounter_id.isin(part.loc[part.partition=='test','encounter_id'])]
   assert set(tr.encounter_id).isdisjoint(te.encounter_id)
   assert set(p.encounter_id)==set(te.encounter_id)
   expected=te.set_index('encounter_id').loc[p.encounter_id]
   assert np.array_equal(p.y,(expected.readmitted=='<30').astype(int))
   assert np.array_equal(p.patient_nbr,expected.patient_nbr)
   seen=np.isin(p.patient_nbr,tr.patient_nbr);assert np.array_equal(seen,p.seen_in_training)
   if arm=='patient':assert not seen.any()
   for _,row in m[(m.arm==arm)&(m.seed==seed)].iterrows():
    values=metrics(p.y,p[row.model],row.threshold)
    for name,value in values.items():assert abs(value-row[name])<1e-10,(seed,arm,row.model,name)
 assert np.isfinite(m[['ap','roc_auc','precision','recall','f1','brier']]).all().all()
 print('PASS: 10 partitions, 60 evaluated predictors, labels, overlap and all reported metrics verified.')
if __name__=='__main__':main()
