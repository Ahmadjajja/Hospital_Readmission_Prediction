# Group walkthrough: what we built and how to explain it

## The question

We are testing whether the way we split repeated hospital records changes the performance we report. We are not inventing a new prediction algorithm.

Imagine patient A has three visits. In an encounter split, two can enter development and one can enter testing. In our primary patient split, all three must stay on the same side. We remove the patient number from model inputs in both cases; grouping and prediction are separate operations.

## What one run does

1. Apply fixed cohort rules and map existing categories. Exclude death/hospice discharges; keep repeated patients.
2. Create a development set and an untouched test set.
3. Divide development into five folds. In the primary protocol, all visits from a patient stay together in every fold.
4. Fit encoders, scalers, class weights and each model only on four training folds. Evaluate the candidate on the remaining fold. Repeat five times.
5. Select the candidate with highest mean validation average precision. Choose a decision threshold from its out-of-fold development predictions.
6. Refit on all development data. Apply the frozen pipeline and threshold to the test set.
7. Save predictions and metrics. Repeat for all five seeds and both protocols.

## Why these models

- L2 logistic regression is the main straightforward baseline. Regularization shrinks coefficients.
- L1 logistic regression can set coefficients to zero, providing embedded feature selection.
- Random forest captures nonlinear rules and interactions by averaging many trees.
- LightGBM sequentially adds trees to improve prediction.
- Unweighted L2 helps reveal how class weighting affects probability quality.
- The constant reference predicts the development prevalence for everyone. It provides a meaningful floor for ranking/probability comparison.

## Metrics in ordinary language

- Average precision: do positive encounters concentrate toward the top of our risk ranking? It is our primary metric because only about 11% of encounters are positive.
- ROC-AUC: how often does a positive case rank above a negative case? Useful alongside AP and prior literature.
- Precision: among cases we flag at the selected threshold, how many are positive?
- Recall: among actual positives, how many do we flag?
- F1: a combined precision/recall score. We select its threshold using development data only.
- Brier: average squared error of probability predictions. Lower is better, but it combines ranking and calibration effects.
- Calibration: when the model says about 30% risk, does about 30% of that group actually have the outcome?

The literal label is `<30`; we use that as positive and combine `>30` with `NO` as negative. We do not reconstruct or redefine dates.

## Why the professor's comments matter

The old diagram could be read as selecting features and oversampling before splitting. That would allow information from later test observations to influence training. The replacement diagram has a separate test branch that never fits anything.

Repeated patients are not duplicate records. We found no exact duplicates, but many patients have multiple distinct encounters. Their visits are correlated; this is why the bootstrap also resamples patients rather than rows.

A missing lab test is not necessarily an unknown lab result. `None` means not performed in the two laboratory-result fields. We keep it as a category. `?` is handled separately as unknown. Default CSV parsing could erase that distinction, so we disable it.

## What our comparison can and cannot establish

The difference between protocol means describes sensitivity to evaluation design. It is not proof that patient overlap alone caused a change, because test populations, class prevalence, tuning folds and chosen settings can also differ. The two protocols address different generalization settings.

The five repeated partitions overlap, so we do not treat the five scores as five independent studies. Their standard deviation shows run variability. The paired patient bootstrap compares models on exactly the same seed-42 patient holdout and is conditional on that fitted run.

## How we differ from Afridi et al.

Patient-level validation and fold-restricted resampling already exist in that paper. Our contribution is the explicit repeated comparison of patient and encounter protocols, overlap measurement and ranking sensitivity, on our documented cohort. We must not claim to invent leakage prevention or directly outperform their different cohort and model suite.

## Completed versus planned

Completed outputs are generated from code and recorded results. SMOTENC, calibration correction, SHAP, demographic subgroup analyses and discharge/transfer sensitivities remain planned. A calibration curve is a diagnostic; it is not a calibration correction. An L1 penalty is implemented feature selection; we have not implemented RFE.

## Review together before submission

- Explain the workflow without reading the code.
- Open `results/metrics.csv` and identify one result in the PDF.
- Explain why test data are never used to select hyperparameters or thresholds.
- Explain why test prevalence matters for AP and seen/unseen comparisons.
- Read the cited clinical papers and check the comparison table.
- Confirm author names and planned responsibilities.
- Review the course's AI-use expectations and disclose assistance as required.

Start with the final report's performance table for the numerical findings. All table values are generated by `generate_report_results.py`; do not manually alter them independently of the result files.
