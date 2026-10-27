"""Does a model know the cat facts of buyers/cat_knowledge.py? Each test is a sentence start with a true and a false
ending; the model "knows" the fact when it gives the true ending the higher probability. Teacher-forced, CPU.

    cd ml && python tools/probe_knowledge.py runs/pretrain_large/final.pt runs/sft_b2/best.pt --out <file.json>

The false endings differ from the true ones by a word or two (often the opposite), so the comparison is about the
fact, not about wording. Topics marked dialogue=False in cat_knowledge.py (breed_pkd, breed_fold) never appear in the
buyer dialogues: a model can only know them from pre-training text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
PAIRS = [  # (topic, sentence start, true ending, false ending)
    ("other_cats", "FIV is a virus that infects", " only cats.", " cats and people."),
    ("other_cats", "FIV spreads mainly through", " deep bite wounds.", " shared food bowls."),
    ("other_cats", "An FIV-positive cat should live", " indoors.", " outdoors."),
    ("other_cats", "A kitten can test FIV positive because of its mother's", " antibodies.", " food."),
    ("other_cats", "A kitten that tests FIV positive is often retested at about", " six months old.", " six years old."),
    ("other_cats", "If the cats in a home do not fight, an FIV-positive cat poses", " little risk to the others.",
     " a high risk to the others."),
    ("other_cats", "Sharing food bowls is", " not an efficient way to spread FIV.", " the main way to spread FIV."),
    ("other_cats", "Before meeting the resident cats, a new cat is usually tested for", " FIV and FeLV.", " fleas and ticks."),
    ("allergies", "Every cat makes Fel d 1, the main", " allergen.", " vitamin."),
    ("allergies", "Neutering a male cat before puberty", " lowers its Fel d 1 level.", " raises its Fel d 1 level."),
    ("allergies", "Compared with females, male cats make", " more Fel d 1.", " less Fel d 1."),
    ("allergies", "An allergic buyer should spend time with the cat", " before adopting it.", " only after adopting it."),
    ("care", "A cat's coat colour says", " nothing about how much it sheds.", " a lot about how much it sheds."),
    ("care", "Long-haired cats need", " regular brushing.", " no brushing at all."),
    ("care", "Vaccines and check-ups are", " recurring costs.", " one-time costs."),
    ("kitten", "Core kitten vaccines start at about", " six to eight weeks.", " six to eight months."),
    ("kitten", "Kitten vaccine boosters are given every", " two to four weeks.", " two to four years."),
    ("kitten", "Vets recommend spaying or neutering a kitten by about", " five months.", " five years."),
    ("kitten", "Spaying a female cat before her first heat lowers the risk of", " mammary cancer.", " kidney stones."),
    ("kitten", "The sensitive socialisation period of a kitten is about", " two to seven weeks of age.",
     " two to seven years of age."),
    ("kitten", "Full protection comes about a week after the", " last kitten vaccine.", " first kitten vaccine."),
    ("kitten", "Rabies vaccination for kittens commonly starts from about", " twelve weeks.", " twelve months."),
    ("young_adult", "A healthy young adult cat usually needs a vet check-up", " once a year.", " every week."),
    ("senior", "Chronic kidney disease is common in", " senior cats.", " young kittens."),
    ("senior", "A sign of chronic kidney disease in an old cat is", " drinking and urinating more.",
     " drinking and urinating less."),
    ("senior", "Hyperthyroidism is a common disorder of", " older cats.", " young kittens."),
    ("senior", "Arthritis in an old cat shows as", " reluctance to jump.", " eagerness to jump."),
    ("senior", "Many cats begin to show age-related changes by about", " ten years of age.", " two years of age."),
    ("senior", "Dental disease is common in", " older cats.", " newborn kittens."),
    ("breed_hcm", "Ragdolls and Maine Coons are predisposed to", " hypertrophic cardiomyopathy.",
     " polycystic kidney disease."),
    ("breed_pkd", "Polycystic kidney disease is common in", " Persian cats.", " Maine Coon cats."),
    ("breed_fold", "The gene that folds a Scottish Fold's ears also causes", " a painful arthritis.", " a thicker coat."),
    ("dogs", "Cats and dogs should be introduced", " slowly.", " all at once."),
    ("dogs", "When a cat first meets a dog, the cat should have", " a dog-free retreat.", " no place to hide."),
    ("outdoors", "Outdoor access exposes a cat to", " more infectious disease.", " less infectious disease."),
    ("outdoors", "A cat used to roaming may struggle to adapt to", " an indoor-only home.", " an outdoor life."),
]


def main() -> None:
    """Score every pair with every checkpoint and write the per-pair log-probabilities and the totals."""
    import torch

    from sellanything_ml.llm.evaluate import load_model
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    report = {}
    for ck in a.checkpoints:
        model, tok = load_model(ck, torch.device("cpu"))

        @torch.no_grad()
        def logp(start: str, end: str) -> float:
            head, full = tok.encode(start), tok.encode(start + end)
            assert full[:len(head)] == head, (start, end)
            logits, _ = model(torch.tensor([full]))
            lp = torch.log_softmax(logits[0, len(head) - 1:len(full) - 1].float(), -1)
            return float(lp.gather(1, torch.tensor(full[len(head):]).unsqueeze(1)).sum())

        rows = [{"topic": t, "start": s, "true": logp(s, y), "false": logp(s, n)} for t, s, y, n in PAIRS]
        report[str(ck)] = {"known": sum(r["true"] > r["false"] for r in rows), "pairs": len(rows),
                           "mean_margin": round(sum(r["true"] - r["false"] for r in rows) / len(rows), 3), "rows": rows}
        print(f"{ck}: {report[str(ck)]['known']}/{len(rows)} known, mean margin {report[str(ck)]['mean_margin']} nats",
              flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
