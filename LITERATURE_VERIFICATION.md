# Literature claim verification

Primary-source methods were checked for the following manuscript claims.

## Afridi et al. (2026)
Source: https://www.mdpi.com/2673-4540/7/10/197 , section 3.1, Dataset Description, and Table 2.
The text reports 75,081 eligible encounters, patient grouping with patient_nbr, and random seed 42. Its preprocessing description confines SMOTENC to training folds, before final one-hot encoding, and leaves validation/test distributions unresampled. These support the manuscript comparison.

## Emi-Johnson and Nkrumah (2025)
Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC12085305/ , Materials and Methods: Study population; Feature engineering and preprocessing; Model development and evaluation.
The methods describe stratified 80/20 splitting, class weighting and five-fold grid search. They do not explicitly establish patient-disjoint splitting in that description. The manuscript therefore says patient grouping is not explicit, and does not claim to have demonstrated leakage in their code.

These checks verify reported methods; they are not independent reproductions of those papers.
