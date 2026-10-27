# Does our cat market point the same way as real adoptions? (2026-10-14)

For the paper. Non-commercial analysis only.

## Data
- PetFinder.my Adoption Prediction (Kaggle competition, 2019), `train.csv`, used under the competition rules
  (7.A: academic research and other non-commercial use; 7.B: not redistributed).
- Kept in `data/external/petfinder/` (excluded from git in `.git/info/exclude`). Never put into training data, prompts or the game.
- Only cats listed one per listing: Type 2, Quantity 1, 4,943 cats from 2,115 rescuers.
- Outcome: not adopted within 100 days (AdoptionSpeed 4).
- Rerun: `python -I ml/tools/petfinder_market_check.py data/external/petfinder/train.csv`

## Method
Logistic regression on age group, vaccinated, sterilised, dewormed, injured, fur length, free and log(1 + photos), run twice:
1. Over all cats, with standard errors clustered by rescuer.
2. Within rescuer (conditional logit), comparing only cats listed by the same rescuer.

The second run follows the 2nd-place solution's GroupKFold by RescuerID: listings by one rescuer share a writing style,
a level of care and a fee policy.

## Results: odds of waiting 100 days unadopted

| Feature | All cats | Within rescuer | Used? |
|---|---|---|---|
| 1 to 2 years (vs 0 to 2 months) | 5.93 | **10.38** | yes |
| 3 years and over | 6.73 | **10.96** | yes |
| long fur (vs short) | 0.30 | **0.31** | yes |
| dewormed | 0.64 | 0.85 (p 0.51) | no: a rescuer effect |
| injured | 1.45 | 1.10 (p 0.84) | no |
| free | 0.63 | 0.69 (p 0.20) | no |
| vaccinated | 1.08 (p 0.58) | 0.83 (p 0.44) | no |
| sterilised | 1.35 | 1.68 | no: the opposite of what a buyer saves; the cause cannot be seen (maybe cats sterilised while waiting) |

Raw share not adopted within 100 days, by age:
- 0 to 2 months: 10.7%
- 3 to 5 months: 20.9%
- 6 to 11 months: 35.4%
- 1 to 2 years: 43.1%
- 3 to 6 years: 44.3%

## What changed in Sell Anything
The owner decided the game only needs a reasonable market, not PetFinder's.

- `ml/tools/asking_price.py`: unchanged.
  - Its age curve already points the same way.
  - The long-haired breeds are priced up as named breeds.
  - The health steps rest on what they cost a buyer, not on this data.
  - A 7-month cat still gets the kitten step. Changing it would rewrite 249 decision lines in 27 written sales for $5.
- Writer instructions (`prompts_2560_2649`, `prompts_2650_2699`): a note that most buyers are more careful with an
  adult's price, each still by their own card.

## Limits
- Malaysia; 83% of cats were free; the median cat was 3 months old.
- Adoption speed is market demand, not one buyer's offer.
- Even the competition's top solutions predicted it only moderately.
- Only age and fur length hold when the rescuer is held fixed.
