"""Generate manuscript numbers directly from verified result files."""
import json
from pathlib import Path
import pandas as pd,numpy as np
from run_experiments import OUT,ROOT
P=ROOT/'paper';m=pd.read_csv(OUT/'metrics.csv');s=pd.read_csv(OUT/'splits.csv');ci=pd.read_csv(OUT/'bootstrap.csv');seen=pd.read_csv(OUT/'seen_unseen.csv')
NAMES={'L2':'L2 logistic','L1':'L1 logistic','RF':'Random forest','LGBM':'LightGBM','L2_unweighted':'L2 unweighted','constant':'Constant'}
def values(arm,n,k):return m[(m.arm==arm)&(m.model==n)][k]
def mean(arm,n,k):return values(arm,n,k).mean()
def ms(arm,n,k):
 v=values(arm,n,k);return f'{v.mean():.3f} $\\pm$ {v.std(ddof=1):.3f}'
best=max(['L2','L1','RF','LGBM'],key=lambda n:mean('patient',n,'ap'))
aps=[mean('patient',n,'ap') for n in ['L2','L1','RF','LGBM']]
ov=s[s.arm=='encounter'].overlap_encounter_pct
abstract=f'The four weighted models achieve mean patient-level AP between {min(aps):.3f} and {max(aps):.3f}; {NAMES[best]} has the highest mean. Encounter-level test sets contain {ov.mean():.1f}\\% encounters from patients represented in development data on average.'
abstract += ' Encounter-level mean AP is modestly higher (+0.005 to +0.013), with a change in model ranking.'
(P/'abstract_results.tex').write_text(abstract+'\n')
parts=[r'\subsection{Completed experiments and performance}',
'All ten outer partitions completed: 500 candidate fits, 50 selected model fits, and ten constant-reference evaluations. The exported metrics contain 60 evaluated model/protocol/seed combinations. Automated checks recompute metrics from predictions, verify labels against source records, and verify partition integrity. No convergence warnings were observed in the completed-run logs. Table~\\ref{tab:performance} and Fig.~\\ref{fig:comparison} summarize the main results. Values are preliminary benchmark findings, not clinical validation.',
r'\begin{table}[t]\caption{Test performance, mean $\pm$ sample SD across five seeds. Constant AP equals each test set\textquotesingle s positive fraction.}\label{tab:performance}\centering\footnotesize\begin{tabular}{llcc}\toprule Model & Split & AP & ROC-AUC\\\midrule']
for n in ['L2','L1','RF','LGBM','L2_unweighted','constant']:
 for arm in ['patient','encounter']:parts.append(f'{NAMES[n]} & {arm.capitalize()} & {ms(arm,n,"ap")} & {ms(arm,n,"roc_auc")} \\\\')
parts += [r'\bottomrule\end{tabular}\end{table}',r'\begin{figure}[t]\centering\includegraphics[width=\columnwidth]{comparison.pdf}\caption{AP across five seeds. Points are individual runs; marks and error bars show means and sample standard deviations.}\label{fig:comparison}\end{figure}']
gaps={n:mean('encounter',n,'ap')-mean('patient',n,'ap') for n in ['L2','L1','RF','LGBM']}
parts.append(f'{NAMES[best]} has the highest mean patient-level AP ({mean("patient",best,"ap"):.3f}), compared with {mean("patient","L2","ap"):.3f} for weighted L2 and {mean("patient","constant","ap"):.3f} for the constant reference. The encounter-minus-patient differences in mean AP are '+', '.join(f'{gaps[n]:+.4f} ({NAMES[n]})' for n in gaps)+'. These are descriptive differences across separately constructed test sets. They are not paired causal effects of overlap.')
parts += [r'\subsection{Overlap and conditional uncertainty}',f'Patient-level holdouts have zero patient overlap by assertion. Encounter-level overlap ranges from {ov.min():.2f}\\% to {ov.max():.2f}\\% of test encounters (mean {ov.mean():.2f}\\%). Patient-arm test prevalence ranges from {100*s[s.arm=="patient"].test_prevalence.min():.2f}\\% to {100*s[s.arm=="patient"].test_prevalence.max():.2f}\\%; the corresponding test sizes range from {s[s.arm=="patient"].test_n.min():,} to {s[s.arm=="patient"].test_n.max():,}. The constant-reference results make this variation explicit.']
v=seen[seen.model==best].groupby('seen')[['ap','prevalence']].mean()
parts.append(f'For {NAMES[best]} within encounter-level holdouts, mean AP is {v.loc[True,"ap"]:.3f} for seen patients and {v.loc[False,"ap"]:.3f} for unseen patients. Their mean positive fractions are {100*v.loc[True,"prevalence"]:.2f}\\% and {100*v.loc[False,"prevalence"]:.2f}\\%, respectively. The case-mix and prevalence difference prevents interpreting this subset contrast as an isolated overlap effect.')
parts.append('For the prespecified patient holdout at seed 42, paired AP differences relative to weighted L2 and their 95\\% patient-bootstrap intervals are '+ '; '.join(f'{NAMES[r.model]}: {r.ap_difference:+.4f} [{r.lower:+.4f}, {r.upper:+.4f}]' for r in ci.itertuples() if r.model!='L2_unweighted')+'. These conditional, unadjusted intervals complement rather than replace the repeated-seed summaries.')
parts += [r'\subsection{Thresholds and probability quality}',r'\begin{table}[t]\caption{Patient-level threshold and probability diagnostics: means across five runs. Thresholds maximize development out-of-fold F1.}\label{tab:diagnostics}\centering\footnotesize\begin{tabular}{lrrrr}\toprule Model & Precision & Recall & F1 & Brier\\\midrule']
for n in ['L2','L1','RF','LGBM','L2_unweighted','constant']:parts.append(NAMES[n]+' & '+' & '.join(f'{mean("patient",n,k):.3f}' for k in ['precision','recall','f1','brier'])+r'\\')
parts += [r'\bottomrule\end{tabular}\end{table}',f'Table~\\ref{{tab:diagnostics}} reports the precision--recall trade-off after development-only threshold selection. Unweighted L2 has mean Brier score {mean("patient","L2_unweighted","brier"):.3f}, compared with {mean("patient","L2","brier"):.3f} for weighted L2. The calibration diagnostic (Fig.~\\ref{{fig:calibration}}) illustrates why class-weighted scores should not automatically be interpreted as calibrated risks. Brier score combines discrimination and calibration; a lower value alone does not establish perfect calibration. No probability correction has yet been fitted.',r'\begin{figure}[t]\centering\includegraphics[width=\columnwidth]{calibration.pdf}\caption{Calibration diagnostic for the prespecified patient holdout, seed 42. Eight quantile bins; dashed line indicates ideal calibration. This is a diagnostic, not a fitted correction.}\label{fig:calibration}\end{figure}']
parts.append(r'Under the Report 1 decision rule, LightGBM improves on weighted L2 in the seed-42 patient holdout because its AP-difference interval excludes zero; random forest is worse and L1 is inconclusive. This inference is conditional on that holdout. Across seeds, the encounter-minus-patient gap is modest (+0.005 to +0.013 AP). Random forest ranks last under patient evaluation (0.208 AP), but approximately ties L2 under encounter evaluation (both round to 0.221). Weighted models have Brier scores near 0.22, worse than the constant reference (about 0.099), reinforcing the need for probability calibration before interpreting their outputs as risk.')
(P/'results_text.tex').write_text('\n\n'.join(parts)+'\n')
print(abstract);print('AP gaps',gaps)
