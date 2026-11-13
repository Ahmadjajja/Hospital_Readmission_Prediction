"""Integrity checks for the actual exported experiments, not smoke tests."""

import json
import numpy as np, pandas as pd
from run_experiments import ROOT, OUT, load, metrics, SEEDS


def main():
    # Basic shape of the results: 10 partitions x 6 predictors, and no IDs or the
    # label among the model inputs.
    df, X, y = load()
    m = pd.read_csv(OUT / "metrics.csv")
    assert len(m) == 60
    assert set(m.seed) == {11, 23, 42, 67, 101}
    assert not {"patient_nbr", "encounter_id", "readmitted", "weight"} & set(X.columns)
    # For every partition, rebuild the split from the saved manifest and check it
    # against the source data.
    for seed in [11, 23, 42, 67, 101]:
        for arm in ["patient", "encounter"]:
            p = pd.read_csv(OUT / f"predictions_{arm}_{seed}.csv.gz")
            part = pd.read_csv(OUT / f"partition_{arm}_{seed}.csv.gz")
            assert part.encounter_id.nunique() == len(df) == len(part)
            tr = df[
                df.encounter_id.isin(
                    part.loc[part.partition == "development", "encounter_id"]
                )
            ]
            te = df[
                df.encounter_id.isin(part.loc[part.partition == "test", "encounter_id"])
            ]
            # Development and test must not share encounters, and the saved predictions
            # must cover exactly the test set.
            assert set(tr.encounter_id).isdisjoint(te.encounter_id)
            assert set(p.encounter_id) == set(te.encounter_id)
            # Labels and patient IDs in the prediction file must match the raw data.
            expected = te.set_index("encounter_id").loc[p.encounter_id]
            assert np.array_equal(p.y, (expected.readmitted == "<30").astype(int))
            assert np.array_equal(p.patient_nbr, expected.patient_nbr)
            # The seen/unseen flag must be correct, and the patient design must have no overlap.
            seen = np.isin(p.patient_nbr, tr.patient_nbr)
            assert np.array_equal(seen, p.seen_in_training)
            if arm == "patient":
                assert not seen.any()
            # Recompute every reported metric from the saved predictions.
            for _, row in m[(m.arm == arm) & (m.seed == seed)].iterrows():
                values = metrics(p.y, p[row.model], row.threshold)
                for name, value in values.items():
                    assert abs(value - row[name]) < 1e-10, (seed, arm, row.model, name)
    assert (
        np.isfinite(m[["ap", "roc_auc", "precision", "recall", "f1", "brier"]])
        .all()
        .all()
    )
    print(
        "PASS: 10 partitions, 60 evaluated predictors, labels, overlap and all reported metrics verified."
    )


if __name__ == "__main__":
    main()
