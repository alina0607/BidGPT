"""Do the cat market's rules (tools/asking_price.py, the buyers' reactions) point the same way as real adoptions?
(owner, 2026-10-14; for the paper, non-commercial analysis only.)

    python -I tools/petfinder_market_check.py ../data/external/petfinder/train.csv

Data: PetFinder.my Adoption Prediction (Kaggle, 2019), train.csv; used under the competition rules (non-commercial
research), kept out of git (.git/info/exclude) and never put into training data or the game. Only cats listed one per
listing (Quantity 1, 4,943 cats), so the adoption speed is that cat's. The outcome is "not adopted within 100 days"
(AdoptionSpeed 4). Two logistic regressions on age group, vaccinated, sterilised, dewormed, injured, fur length, free,
log(1 + photos): one over all cats with standard errors clustered by rescuer, one within rescuer (conditional logit:
only cats listed by the same rescuer are compared), following the 2nd-place write-up's GroupKFold by RescuerID. A
feature counts as supported only when both agree. Results 2026-10-14: age (1-2 years 10.4x, 3 years and over 11.0x the
odds of a 0-2 month kitten, within rescuer) and long fur (0.31x) hold; dewormed, injured and free do not hold within
rescuer; vaccinated shows nothing; sterilised points the "wrong" way in both (1.35x, 1.68x) for a reason the data
cannot show, so it is not used.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def main() -> None:
    import statsmodels.formula.api as smf
    from statsmodels.discrete.conditional_models import ConditionalLogit
    df = pd.read_csv(sys.argv[1])
    c = df[(df.Type == 2) & (df.Quantity == 1)].copy()
    c["age"] = pd.cut(c.Age, [-1, 2, 5, 11, 35, 400], labels=["a0_2m", "a3_5m", "a6_11m", "a1_2y", "a3y_plus"]).astype(str)
    for k, col in (("vacc", "Vaccinated"), ("ster", "Sterilized"), ("dew", "Dewormed")):
        c[k] = c[col].map({1: "yes", 2: "no", 3: "unsure"})
    c["injured"] = (c.Health > 1).astype(int)
    c["fur"] = c.FurLength.map({1: "short", 2: "medium", 3: "long"})
    c["free"] = (c.Fee == 0).astype(int)
    c["photos"] = np.log1p(c.PhotoAmt)
    c["not_adopted"] = (c.AdoptionSpeed == 4).astype(int)
    print(f"{len(c)} single-cat listings from {c.RescuerID.nunique()} rescuers")
    order = ["a0_2m", "a3_5m", "a6_11m", "a1_2y", "a3y_plus"]
    print(c.groupby("age").not_adopted.agg(["size", "mean"]).reindex(order).round(3).to_string())
    f = ("not_adopted ~ C(age, Treatment('a0_2m')) + C(vacc, Treatment('no')) + C(ster, Treatment('no')) + "
         "C(dew, Treatment('no')) + injured + C(fur, Treatment('short')) + free + photos")
    m1 = smf.logit(f, c).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(c.RescuerID)[0]})
    x = pd.get_dummies(c[["age", "vacc", "ster", "dew", "fur"]]).astype(float)
    x = x.drop(columns=["age_a0_2m", "vacc_no", "ster_no", "dew_no", "fur_short"])
    x["injured"], x["free"], x["photos"] = c.injured, c.free, c.photos
    m2 = ConditionalLogit(c.not_adopted.values, x.values, groups=c.RescuerID.values).fit(disp=0)
    within = pd.DataFrame({"odds": np.exp(m2.params), "p": m2.pvalues}, index=x.columns)
    pairs = {"age_a1_2y": "C(age, Treatment('a0_2m'))[T.a1_2y]", "age_a3y_plus": "C(age, Treatment('a0_2m'))[T.a3y_plus]",
             "vacc_yes": "C(vacc, Treatment('no'))[T.yes]", "ster_yes": "C(ster, Treatment('no'))[T.yes]",
             "dew_yes": "C(dew, Treatment('no'))[T.yes]", "injured": "injured",
             "fur_long": "C(fur, Treatment('short'))[T.long]", "free": "free"}
    print(f"\n{'feature':14s}{'all (rescuer-clustered)':>28s}{'within rescuer':>22s}")
    for short, long in pairs.items():
        print(f"{short:14s}{np.exp(m1.params[long]):12.2f}  p={m1.pvalues[long]:.3f}"
              f"{within.odds[short]:12.2f}  p={within.p[short]:.3f}")


if __name__ == "__main__":
    main()
