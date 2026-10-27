# DomainNet sketch: hand review of cat, dog and bird (2026-10-07)

Source: DomainNet (Peng et al., ICCV 2019), sketch domain, `sketch.zip` (2,633,992,926 bytes, SHA-256 in
`ml/data/raw/sketch_extra/SHA256SUMS`) and the authors' cleaned split lists `sketch_train.txt` / `sketch_test.txt`
(48,212 + 20,916 = 69,128 images, 345 classes; the zip holds 70,386 files, the lists leave out 1,258).
Licence: non-commercial research only.

Every listed image of the three classes (cat 130, dog 310, bird 306 = 746) was looked at, 16 per sheet at
300 px. `review.tsv` holds one row per image: path, class, split, decision, reason.

## Rules
Keep (`K`): one hand-made drawing of the class (whole animal or a part such as the head), in pencil, pen,
ink or digital line, on paper or a plain background. Allowed: the artist's signature, a date, a short
handwritten note, a small artist credit, coloured *lines*, tinted paper, a scene in which the animal is the
subject (two cats curled together, a bird on a branch with flowers).

Reject:

| Code | Meaning |
|---|---|
| `W` | a stock-site or web watermark, logo, URL bar or footer (dreamstime, shutterstock, 123RF, CanStock, wikiHow, DragoArt, …) |
| `M` | more than one separate drawing: several animals drawn apart, study sheets, step sequences, the same drawing twice |
| `P` | a photograph or real objects in the picture: pencils, hands, tables, tattoos on skin, photographed paper with non-paper surroundings |
| `T` | printed or large text that is part of the picture: titles, tutorial captions, cards, paragraphs of notes |
| `F` | an added frame or border, or the transparency checkerboard |
| `N` | not a sketch: filled colour painting, colour illustration, logo, silhouette, white-on-black |
| `C` | the class is not the subject (people walking a dog, a birdcage) |
| `S` | a screenshot of an app, a web page or a paper figure |
| `E` | empty or unusable (black image, a smudge) |

`N` was added while reviewing the dogs (dog 90); the 130 cats were reviewed before it existed. Looking back
at the cat sheets, no cat was kept that `N` would reject.

## Result

| | kept / all | train kept | test kept |
|---|---|---|---|
| cat | 63 / 130 | 42 / 91 | 21 / 39 |
| dog | 136 / 310 | 96 / 217 | 40 / 93 |
| bird | 122 / 306 | 86 / 214 | 36 / 92 |
| **all** | **321 / 746 (43%)** | **224** | **97** |

Reasons for rejection: watermark 186, several drawings 86, photo or objects 55, not a sketch 39,
text 23, frame 15, class not the subject 9, screenshot 7, empty 5.

## Limits
- Faint stock watermarks are hard to see even at 300 px. Two (dog 149, dog 151) were found only by opening
  the image at full size, so some kept images may still carry a faint watermark.
- The rules are one reviewer's judgement; borderline cases (a tiny second doodle, handwritten notes) were
  decided by the rule text above, and the reason is written down for every rejection.
- The other 342 classes were not reviewed. They are used only as DomainNet's own cleaned lists.
