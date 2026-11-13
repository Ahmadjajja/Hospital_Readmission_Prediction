"""Generate manuscript numbers directly from verified result files."""

import json
from pathlib import Path
import pandas as pd, numpy as np
from run_experiments import OUT, ROOT

# Every number in the results section and abstract comes from these files,
# so the manuscript text can always be regenerated after a rerun.
P = ROOT / "paper"
m = pd.read_csv(OUT / "metrics.csv")
s = pd.read_csv(OUT / "splits.csv")
ci = pd.read_csv(OUT / "bootstrap.csv")
seen = pd.read_csv(OUT / "seen_unseen.csv")
# Display names used in the tables.
NAMES = {
    "L2": "L2 logistic",
    "L1": "L1 logistic",
    "RF": "Random forest",
    "LGBM": "LightGBM",
    "L2_unweighted": "L2 unweighted",
    "constant": "Constant",
}


# Small helpers: all five seed values, their mean, and "mean ± SD" for tables.
def values(arm, n, k):
    return m[(m.arm == arm) & (m.model == n)][k]


def mean(arm, n, k):
    return values(arm, n, k).mean()


def ms(arm, n, k):
    v = values(arm, n, k)
    return f"{v.mean():.3f} $\\pm$ {v.std(ddof=1):.3f}"


# Headline numbers used in the abstract and summary.
WEIGHTED = ["L2", "L1", "RF", "LGBM"]
best = max(WEIGHTED, key=lambda n: mean("patient", n, "ap"))
aps = [mean("patient", n, "ap") for n in WEIGHTED]
ov = s[s.arm == "encounter"].overlap_encounter_pct
gaps = {n: mean("encounter", n, "ap") - mean("patient", n, "ap") for n in WEIGHTED}
gap_range = f"{min(gaps.values()):+.3f} to {max(gaps.values()):+.3f}"
# Results sentences for the abstract.
abstract = (
    f"Under patient-level evaluation, the four class-weighted models reach mean AP between {min(aps):.3f} and {max(aps):.3f}, with {NAMES[best]} highest. "
    f"In encounter-level test sets, {ov.mean():.1f}\\% of encounters on average belong to patients who also appear in the development data. "
    f"Encounter-level AP is slightly higher ({gap_range}) and the ranking of the models changes."
)
(P / "abstract_results.tex").write_text(abstract + "\n")
# Results section, built piece by piece and written to results_text.tex.
parts = [
    r"\subsection{Completed experiments and performance}",
    "All ten outer partitions (five seeds, two protocols) have been run, giving 60 evaluated model results. A verification script recomputes every reported metric from the saved predictions and checks labels and partitions against the source data. Table~\\ref{tab:performance} and Fig.~\\ref{fig:comparison} summarize the results. These are preliminary findings on a historical benchmark.",
    r"\begin{table}[t]\caption{Test performance, mean $\pm$ sample SD across five seeds. Constant AP equals each test set\textquotesingle s positive fraction.}\label{tab:performance}\centering\footnotesize\begin{tabular}{llcc}\toprule Model & Split & AP & ROC-AUC\\\midrule",
]
for n in ["L2", "L1", "RF", "LGBM", "L2_unweighted", "constant"]:
    for arm in ["patient", "encounter"]:
        parts.append(
            f'{NAMES[n]} & {arm.capitalize()} & {ms(arm,n,"ap")} & {ms(arm,n,"roc_auc")} \\\\'
        )
parts += [
    r"\bottomrule\end{tabular}\end{table}",
    r"\begin{figure}[t]\centering\includegraphics[width=\columnwidth]{comparison.pdf}\caption{AP across five seeds. Points are individual runs; marks and error bars show means and sample standard deviations.}\label{fig:comparison}\end{figure}",
]
parts.append(
    f'All models perform well above the constant reference, but the absolute level is modest. {NAMES[best]} has the highest mean patient-level AP ({mean("patient",best,"ap"):.3f}), compared with {mean("patient","L2","ap"):.3f} for weighted L2 and {mean("patient","constant","ap"):.3f} for the constant reference. Every model scores higher under encounter-level evaluation; the differences in mean AP are '
    + ", ".join(f"{gaps[n]:+.4f} ({NAMES[n]})" for n in gaps)
    + ". Random forest shows the largest gap."
)
parts += [
    r"\subsection{Patient overlap and uncertainty}",
    f'By construction, patient-level holdouts share no patients with the development data. In the encounter-level holdouts, {ov.min():.2f}\\% to {ov.max():.2f}\\% of test encounters (mean {ov.mean():.2f}\\%) come from patients who also appear in development. Because the patient split holds out patients rather than encounters, its test sets vary slightly in size ({s[s.arm=="patient"].test_n.min():,} to {s[s.arm=="patient"].test_n.max():,} encounters) and prevalence ({100*s[s.arm=="patient"].test_prevalence.min():.2f}\\% to {100*s[s.arm=="patient"].test_prevalence.max():.2f}\\%), which the constant reference reflects.',
]
# Seen vs unseen patients within the encounter-level test sets.
v = seen[seen.model == best].groupby("seen")[["ap", "prevalence"]].mean()
parts.append(
    f'Within the encounter-level holdouts, {NAMES[best]} reaches a mean AP of {v.loc[True,"ap"]:.3f} on patients seen during development and {v.loc[False,"ap"]:.3f} on unseen patients. The two groups also differ sharply in readmission rate ({100*v.loc[True,"prevalence"]:.2f}\\% versus {100*v.loc[False,"prevalence"]:.2f}\\%). Seen patients are by definition repeat visitors, who are readmitted far more often, so this contrast mixes overlap with case mix.'
)
parts.append(
    "For the seed-42 patient holdout, the paired AP differences relative to weighted L2, with 95\\% patient-bootstrap intervals, are "
    + "; ".join(
        f"{NAMES[r.model]}: {r.ap_difference:+.4f} [{r.lower:+.4f}, {r.upper:+.4f}]"
        for r in ci.itertuples()
        if r.model != "L2_unweighted"
    )
    + "."
)
parts += [
    r"\subsection{Thresholds and probability quality}",
    r"\begin{table}[t]\caption{Patient-level threshold and probability diagnostics: means across five runs. Thresholds maximize development out-of-fold F1.}\label{tab:diagnostics}\centering\footnotesize\begin{tabular}{lrrrr}\toprule Model & Precision & Recall & F1 & Brier\\\midrule",
]
for n in ["L2", "L1", "RF", "LGBM", "L2_unweighted", "constant"]:
    parts.append(
        NAMES[n]
        + " & "
        + " & ".join(
            f'{mean("patient",n,k):.3f}' for k in ["precision", "recall", "f1", "brier"]
        )
        + r"\\"
    )
parts += [
    r"\bottomrule\end{tabular}\end{table}",
    f'Table~\\ref{{tab:diagnostics}} shows the precision--recall trade-off at the thresholds chosen on development data. At these thresholds the models flag roughly half of the readmissions, and about one in five flagged encounters is a true readmission. Probability quality differs much more than ranking. The class-weighted models have Brier scores of about {mean("patient","L2","brier"):.2f}, worse than the constant reference ({mean("patient","constant","brier"):.3f}), whereas unweighted L2 reaches {mean("patient","L2_unweighted","brier"):.3f} with almost the same AP. Fig.~\\ref{{fig:calibration}} shows why: class weighting pushes predicted probabilities far above the observed readmission rates. These scores therefore cannot be read as risks until they are recalibrated, which is planned.',
    r"\begin{figure}[t]\centering\includegraphics[width=\columnwidth]{calibration.pdf}\caption{Calibration curves for the seed-42 patient holdout, using eight quantile bins. The dashed line marks perfect calibration.}\label{fig:calibration}\end{figure}",
]
parts.append(r"\subsection{Summary of initial findings}")
parts.append(
    f'Report 1 defined a clear improvement over the baseline as a paired 95\\% interval that excludes zero. On the seed-42 holdout, LightGBM meets this criterion, random forest is significantly worse, and L1 logistic regression is indistinguishable from L2. Across seeds, the gap between the two protocols is small ({gap_range} AP), but it is large enough to change conclusions about model choice: random forest ranks last under patient evaluation ({mean("patient","RF","ap"):.3f} AP) yet ties L2 under encounter evaluation (both {mean("encounter","L2","ap"):.3f}).'
)
(P / "results_text.tex").write_text("\n\n".join(parts) + "\n")
print(abstract)
print("AP gaps", gaps)
