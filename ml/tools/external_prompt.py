"""Generation prompts for writing cat sales with another model (owner, 2026-10-12: Muse), one per seed.

    PYTHONPATH=. python tools/external_prompt.py 2100 2199 ../data/external_sales/muse/prompts
    PYTHONPATH=. python tools/external_prompt.py 2400 2499 ../data/external_sales/muse_v3/prompts --quotas

With --quotas (the second Muse batch, 2026-10-13) each prompt also carries the rules learned from reading and fixing
the first 100 sales, and a list of drawable things this sale's scenes should use, taken from the things the sales so
far picture least (tools/scene_quota.py: a thing seen fewer than 10 times is not learned, 30 is safer).
With --human (seeds 2650-2699, NEXT_SESSION_PROMPT §4) each prompt also asks for a seller who writes like a real
player, and names the two extra facts stories.sale_for_seed adds to that cat. Seeds in stories.LIFE_SEEDS also get the
life-facts section (what real adopters ask: origin, being picked up, food, fleas, microchip, litter); --append-life adds or replaces it
in prompts already written in the folder without touching their other text:
    PYTHONPATH=. python tools/external_prompt.py 2593 2649 ../data/written_sales/prompts_2560_2649 --append-life

Each prompt holds that seed's cat, seller style and ten buyers (stories.seed_sale), the general cat facts for the
topics they touch, every rule the training sales are held to, and the output format tools/write_sales.py reads. The
answer for seed N is saved as <specs_dir>/N.json and checked with `python tools/write_sales.py <specs_dir> <out>`.
Rules added after reading Muse's first sale (devlog 2026-10-12 §7) are marked as such in the text.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from sellanything_ml.buyers import stories as S


QUOTA_FILE = Path(__file__).resolve().parents[2] / "data" / "written_sales" / "scene_quota.json"
LISTED = 12                        # quota things offered to one sale
OVERUSED = 150                     # things pictured this often already are not used in these sales (except dog)


def quota_lists(seeds: list[int], target: int = 30) -> tuple[dict[int, list[str]], list[str]]:
    """For every seed, LISTED home things to picture, drawn in proportion to how far each is from `target` scenes
    (three times as likely below 10), and lowered as they are handed out (about 1.5 picks per listing); and the
    overused things. Once nearly every thing has 30 scenes the lists run dry, so later seeds use target 50 (2026-10-26,
    seeds 2737 to 2799)."""
    q = json.loads(QUOTA_FILE.read_text())
    left = {t: max(0, target - n) * (3 if n < 10 else 1) for t, n in q["counts"].items() if t not in q["not_home"]}
    rng = random.Random(seeds[0])
    out = {}
    for seed in seeds:
        pool = [t for t, w in left.items() if w > 0]
        chosen: list[str] = []
        while len(chosen) < LISTED and pool:
            t = rng.choices(pool, weights=[left[x] for x in pool])[0]
            chosen.append(t)
            pool.remove(t)
        for t in chosen:
            left[t] = max(0, left[t] - (4.5 if q["counts"][t] < 10 else 1.5))
        out[seed] = sorted(chosen)
    overused = sorted(t for t, n in q["counts"].items() if n >= OVERUSED and t != "dog")
    return out, overused


def lessons(listed: list[str], overused: list[str]) -> str:
    """The rules learned from fixing the first 100 Muse sales, and this sale's scene things."""
    return f'''
## Rules learned from the first 100 sales written for this project (each was a common fault; every one is checked)

**The seller**
- States only the facts above. Anything else (how she gets on with others, naps, size, coat colour, eye colour, past
  owners, a vet's opinion, how she will behave in a new home) is "I don't know" or left out; never a guess.
- Never claims the cat's sex. Pick one pronoun and keep it; no buyer asks or claims the sex either.
- Every fact on the sheet comes out over the rounds, each in the round whose question it answers; do not volunteer a
  fact no buyer asked about.

**Moves (about 60 to 65 out of 100 reactions keep the offer; the rest move)**
- For each answer ask: does it cost this buyer money, time, sleep, health, space, or a hope they named? Then LOWER.
  Does it give them something their card says they need? Then RAISE. Only when neither, KEEP.
- A new cost (shots, spay or neuter, checkups, a special diet, a fountain) lowers a buyer who works for a living,
  once, naming the cost. "My pay covers it", "my pension covers it", "it's a small bill", free days or "routine for
  any cat" is NOT a reason to keep: free time or income never cancels a cost. Keeping is fine only with a real
  reason: a fountain already owned, or the cost already counted in an earlier round.
- A plus that fits the buyer's card is a raise, never a keep. A drawback that merely does not bother them (a loud
  breakfast call when they are up anyway, brushing, shyness) is a keep, never a raise.
- A buyer who is out all day (office, ward, shop, school) lowers once for "used to company", "follows people" or
  "very cuddly", and needs a fountain for a cat that only drinks from a running tap. A buyer with dogs or an older
  cat at home all day is not leaving the cat alone, so keeps.
- An unknown ("I don't know how much she sheds") is a keep, and the buyer says how they will find out; only an
  unknown that their card is about (dogs for a dog owner, children for a parent) may lower once.
- One worry is counted once, even when a later answer touches it again. Never lower twice for the same fact.
- Withdraw only when the cat can never work for this buyer (a breeder hearing no papers or spayed; a hiker hearing
  an adult cat is FIV positive and must stay in). Never withdraw over a bill; an indoor-only cat is a lower for a
  hiker, not a withdrawal.

**How buyers talk**
- Each line names what the seller just said and ties it to this buyer's own life from their card and greeting;
  12 to 30 words, one or two sentences. Every line of one buyer agrees with that buyer's job and home.
- Vary the endings: "Down to $N.", "Up to $N.", "I'll hold at $N.", "$N still stands.", "I'm keeping my $N.", or the
  amount inside the sentence. At most one line in five ends "Staying at $N." or "Holding at $N."
- Overused, do not use: "doesn't show in a photo/portrait" (at most once per sale), "X is temperament, not
  condition", "X doesn't apply to me", "pocket change".
- Persona details that already repeat across sales, do not use: twins; a flat above a bakery; a cardiac ward;
  restoring watches; retired from the post office or railway; cream, brass or marble rooms; pets named Pepper,
  Biscuit, Mochi, Miso or Pickle; Labradors or retrievers. Give each buyer one specific detail IN THEIR GREETING
  instead (an unusual job, a hobby, the ages of the kids, a different dog breed). Reactions and scenes only use what
  the card, the greeting or the buyer's own question already said: no new street, plan, habit or chore later.
- Greetings are fresh for this sale, and no greeting knows a fact before the seller says it (no breed, colour or
  age until it is said).

**Scenes (this matters most for the model)**
- Each scene line is short: at most 25 words, about one clause per thing.
- Over all the scenes of this sale, use THIS SALE'S LIST below at least 14 times (no list thing more than twice),
  but only where a thing truly fits what the buyer's card and greeting already say. Never invent a new detail of a
  buyer's life (a windmill they own, a diamond they cut) to make a list thing fit: give that buyer other things.
- This sale's list: {", ".join(listed)}.
- Do not use these overused things at all in this sale: {", ".join(overused)}.
- Cooked plain fish or chicken only; never human food for the cat, never raw fish. An indoor cat is pictured indoors.
'''


HOW_BUYERS_TAKE = {   # NEXT_SESSION_PROMPT §4.1 A: how each new kind of fact moves a buyer
    "very old": 'The age is read exactly as said: "20" or "20 yrs" is twenty years, never two. Senior care and fewer '
                "years together are costs; a buyer who wants a young cat, plays or breeds treats it as the age it is.",
    "heart": "A heart problem (a murmur, HCM, an episode last year) is a serious long-term cost and risk: "
             'checkups and an uncertain future (pills only when the seller said so; no guessed treatments). It is never good news, and "the vet says stable now" does not make '
             "it fine. Every buyer lowers once for it (free time or a pension does not cancel it); a breeder withdraws.",
    "no papers": '"Pure breed but no papers" is two facts: the breed is the seller\'s word, the papers are missing. A '
                 "breeder or show buyer cannot register the cat and withdraws; a buyer who prizes pedigree lowers "
                 "once for the missing proof; everyone else reacts to the breed as they would and keeps for the papers.",
    "diabetes": "An insulin shot every day is a daily duty at fixed times and a recurring cost: lower once; a buyer "
                "out all day or on night shifts says what the fixed times mean for them.",
    "thyroid": "Pills every day for the thyroid are a recurring cost: lower once.",
    "one eye": "Blind in one eye: the seller said nothing about how the cat copes, so buyers do not guess (no "
               '"bumps into things"). Keep and say how they will see it move, unless the card names a real cost.',
    "three legs": "Three legs: the seller said nothing about pain or climbing, so buyers do not guess. Keep and say "
                  "how they will see it move, unless the card names a real cost (an adventure or hiking buyer may "
                  "lower once).",
    "deaf": "Born deaf: a deaf cat cannot hear cars or a dog coming, which matters on a busy road or outdoors (lower "
            "once there); indoors most buyers keep.",
    "teeth": "Teeth pulled last year is done and paid for: most buyers keep; it is not a new cost.",
}


def human(seed: int) -> str:
    """The more-human-seller section (NEXT_SESSION_PROMPT §4.1 A to C and the §4.2 rules), with this cat's new facts."""
    c = S.sale_for_seed(seed)["cat"]
    new = "\n".join(f"- {f}" for f in c["extra"])
    take = "\n".join(f"- {HOW_BUYERS_TAKE[k]}" for k in c["extra_kinds"])
    return f'''
## This batch: a seller who talks like a real player (read carefully; every point is checked or read)

Why: in play, the buyer model misread real players. "Heart attack" was taken as a good sign, "20 years old" was read
as 2, and "she is pure breed but no paper" as "no papers and not pure bred". Sellers in the training sales only gave
tidy facts from a fixed sheet. This batch teaches the buyers to read how people really write.

**A. Two new facts on this cat's sheet** (they are true facts like the others):
{new}
Both come out, each in the answer to a question it truly answers. How buyers take them:
{take}

**B. The seller talks like a real person.** Over the sale, at least:
- three answers that join two things with "but" or "though" ("pure breed though no papers sry",
  "had a heart thing last yr but vet says stable now", "shes 20 but bloodwork was fine last month");
- two answers with a lowercase start or a casual typo (shes, dont, sry, yr, vaccs, prob);
- one number written in digits ("20", "18 yrs", "2 of 3 shots").
The examples are from other cats: use this cat's own facts. The seller keeps their way of answering, still says
only facts from the sheet (never adds one to make a "but" sentence), never infers, and never says the cat's sex. The note stays in plain, correct words.

**C. Buyers read the whole answer.** When an answer holds two things, each buyer's reaction answers every part that
concerns them (both halves of "pure breed but no papers"). Numbers and ages are read exactly as written.

**Rules fixed while writing the last batch** (2026-10-13; every one was a real fault):
- Unknowns (papers, tests, history) keep the offer, and the buyer says how they will find out, breeders too; never
  lower for what a test might show. A fact that makes the buyer's purpose impossible (a breeding line with a very old
  cat) is a withdrawal, not a cut.
- A fact answered earlier is never a reason again, not even as a side clause; a worry already counted is "already
  in my offer" and keeps.
- A new cost lowers once; different costs lower once each; a plus raises once. A cost never rides inside a raise and
  a plus never inside a cut.
- A hiker or outdoor buyer who hears indoor-only lowers.
- A raise or cut comes only from what the seller said or the general cat facts, never the buyer's own guess of
  whether the cat is easy to keep.
- When a buyer's own question said why they ask ("early neutering lowers allergens"), their reaction follows that
  reason: an allergic buyer hearing a kitten is not yet neutered keeps (it can be done before puberty).
- No invented preference waves a cost away. Never "what I want", "what I wanted" or "I'd hoped".
- Within a round, buyers who keep never share one reason sentence.
- A kitten's remaining booster shots are a cost, never a raise for being "on schedule".
- A buyer never contradicts what they said earlier. Moves are $5 to $15.
- "Out all day" and where the cat would live come only from the card or the greeting.
- Scenes at most 25 words, from the card and the greeting. No dash anywhere; never mention AI or any assistant.
'''


def prompt(seed: int, extra: str = "") -> str:
    s = S.sale_for_seed(seed)
    buyers = "\n".join(f"{b['name']}: {b['persona']}" for b in s["buyers"])
    things = ", ".join(sorted(S.DRAWABLE_THINGS))
    vague = "; ".join(f'"{v}"' for v in S.VAGUE)
    return f'''# Task: write one complete cat sale (seed {seed}) as training data for a small buyer model

You are writing training data for a small language model that plays the BUYERS in a game. In the game the player
sells a cat at a secondhand market; buyers ask the seller questions, react to each answer by raising, lowering or
keeping their offer with a spoken reason, and at the end each buyer pictures the cat at their home. Write ONE
complete sale for the cat and buyers below, as a single JSON object in the exact format at the end, and nothing else.
Quality matters far more than speed: every line is checked by a program and read by a person.

## The setting
A secondhand marketplace. A typical price for a cat here is about $75. Buyers have NO spending limit; their offers
only have to move reasonably (see the rules).

## The cat (true facts; only the seller knows them; buyers learn only what the seller says)
{S.facts(s)}.

Seller's way of answering: {s["seller_style"]}.

## The ten buyers (name: who they are)
{buyers}

## True facts about cats in general (everyone may use them; nobody may contradict them)
{S.knowledge(s)}

## What to write

1. **Cards.** For each buyer: `cares_about`, two or three short phrases saying what they want IN A CAT, following
   from who they are (e.g. "used to dogs", "low allergen", "quiet in the day"), not a restatement of their life
   ("retired", "has two dogs"). No budget.
2. **Openings.** For each buyer: a starting offer in whole dollars (around $75; a premium or pedigree buyer may start
   higher, a deal-hunter or student lower) and a greeting of one or two sentences that tells one concrete thing
   about their own life explaining what they want. Not the formula "I am X so I want Y". The greetings differ in
   length and tone.
3. **Rounds** (up to ten; each buyer asks at most once; a buyer who has withdrawn never asks again). In each round:
   - `name`: the buyer who asks; `q`: the question that buyer most needs answered, given their card and what has
     already been said. Never ask again about something already asked or answered, even in other words. Cover
     different topics (health, age, temper, other cats, dogs, children, indoors or outdoors, grooming and shedding,
     allergies, costs, daily routine, noise, breed and papers, ...), and vary how people ask. A question is a full
     sentence of at least five words, the way a person asks at a market ("Health. The full picture." is not).
   - `a`: the seller's answer, one to three sentences, in the seller's way of answering, answering the question
     asked. True to the cat's facts: the seller may be vague or play things up if that is their style, but never
     states something false as a fact, never contradicts the general cat facts, and never hints at a risk those facts
     rule out (for example telling someone with kids or dogs to be careful because the cat is FIV positive). The
     seller only knows the facts above; anything else (prices, papers, history not listed) is "I don't know" or
     vague.
   - `note`: what a buyer would write down from this answer: the facts it gave, at most 12 words, keeping every
     "never"/"not"/"no", and nothing the seller did not say.
   - `react`: one reaction from EVERY buyer still bidding (including the asker), as `[sign, offer, says]`.
4. **Imagine** (after the last round): for every buyer still bidding, `[[thing1, thing2, thing3], says]`.

## Rules for reactions (each one is checked)
- React to the LATEST answer only. A fact from an earlier answer is already in the buyer's offer and is NEVER the
  reason again, even when the seller mentions it again or a new answer touches it: a buyer who lowered for vet costs
  does not lower again when costs come up later. (Added after the first sale: buyers lowered three times for the
  same fact.)
- `sign` is "+" (offer goes up), "-" (offer goes down, or 0 to withdraw) or "0" (offer stays exactly the same).
- `says` is what the buyer says aloud: one or two fluent first-person sentences of at least eight words, the way a
  real person talks at a market (not clipped notes like "Health is value. Holding at $68."). It names what the seller just said AND how that meets or misses the buyer's own concrete situation (their
  work, home, family, other pets, health, hobbies, money), as their card describes it. If the card does not mention
  dogs, children or another cat, the buyer has none.
- Every reason is true and sensible: living indoors is safer for a cat, not a health minus; a cat that will live
  in the same home as a resident cat can still fight with it indoors. (Added after the first sale.)
- When the offer changes, `says` ends with the new amount ("Up to $80." / "Down to $60."), and that amount equals
  `offer`. When it stays, `says` may state the same amount.
- An unchanged offer must still name what the seller just said, reusing at least one word of four or more letters
  from the answer, and say in the buyer's own words why it does not matter for them. Within one round, no two
  buyers say the same sentence or the same sentence shape (not six lines of "X doesn't apply to me"), and no three
  buyers start with the same three words. (Added.)
- Every greeting, question and reaction is written for this sale: never reuse a line from another sale, even with
  another amount. (Added after the first 100 sales: 501 reaction lines appeared in two or more sales.)
- Moves are reasonable: a raise or cut is usually $5 to $15 and never more than 40% of the current offer; a buyer's
  offer never goes above 1.6 times their opening offer.
- Withdraw (sign "-", offer 0) only when something the seller actually said makes this cat impossible for this buyer
  (a breeder hearing no papers or spayed; a hiker hearing the cat must stay indoors; a student whose costs are too
  high), and the line names what was said and why it rules the cat out. Withdrawing is final.
- Every reason agrees with the general cat facts (FIV cannot pass to people or dogs; a kitten's FIV result may be
  its mother's antibodies; coat colour says nothing about shedding; no cat is hypoallergenic).
- Never: a dash (— or –); the buyer saying their own name; a sentence starting "She being", "He being" or "The cat
  being"; any of these empty phrases: {vague}.

## Rules for the imagined scene (each one is checked)
- Exactly three things, each spelled exactly as in this list of things that can be drawn (not "cat"):
  {things}.
- Things from the buyer's own life (work, hobbies, family, home) as the card describes it; not furniture anyone has
  (no plain chair, pillow, couch or bed unless it says something about this buyer). No two buyers share two or more
  things.
- The scene agrees with everything the seller said: a cat that must stay indoors is pictured indoors (no barn,
  fence, garden or trail). (Added after the first sale.)
- `says` (one or two first-person sentences, at least twelve words, with "I" or "my") names every thing picked and
  what it is for in the buyer's life with this cat, and mentions no other thing from the list and none of these words (they cannot be drawn):
  {", ".join(S.NOT_DRAWABLE)}.

## Output: one JSON object, nothing else

{{"cards": {{"<name>": {{"cares_about": ["...", "..."]}}, "...": "every buyer"}},
 "open": {{"<name>": [70, "greeting"], "...": "every buyer"}},
 "rounds": [{{"name": "<asker>", "q": "...", "a": "...", "note": "...",
             "react": {{"<name>": ["0", 70, "..."], "...": "every buyer still bidding"}}}}],
 "imagine": {{"<name>": [["thing", "thing", "thing"], "..."], "...": "every buyer still bidding"}}}}

A short example of the style (a different cat: 13 years old, FIV positive, an honest seller):
  Sam asks "She's thirteen. How is her health?"
  answer: "Her bloodwork last month was normal. I'll be honest though: she tested FIV positive. She has no symptoms,
           but she'll need a checkup every six months."
  note: "bloodwork normal; FIV positive; no symptoms; checkup every six months"
  Sam (wants excellent health): ["-", 55, "Normal bloodwork is good, but FIV positive at thirteen is not the
           excellent shape I want. Down to $55."]
  Felix (retired, home all day): ["+", 60, "Normal bloodwork at thirteen, and I'm home for a checkup every six
           months. Up to $60."]
  Ben (has an old cat, Max): ["-", 40, "Testing FIV positive worries me with old Max at home. Down to $40."]
  imagine, Felix: [["couch", "television", "coffee cup"], "She'd chat at me from the couch while the television hums
           and my coffee cup cools."]
Write your own lines; do not copy the example.
{extra}'''


LIFE_ASK = {   # the question that brings each kind of life fact out, and how buyers take the answer
    "origin": ("why the cat is being rehomed or where the cat came from",
               "Where a cat came from is history, not a cost: buyers keep. A stray or unknown past is an unknown: keep "
               "and say how you will find out (a vet check)."),
    "handling": ("whether the cat likes being picked up or held",
                 "A buyer whose card or own words need a cat to be held (young children, wants a lap cat) raises once "
                 "for one who loves it and lowers once for one who hates it; everyone else keeps."),
    "food": ("what the cat eats now",
             "A set diet is not a cost on its own: the new owner keeps the same food and changes it slowly over about "
             "a week. A buyer on a tight budget may lower once for a cat that only eats wet food; others keep."),
    "pests": ("whether the cat has had flea or ear mite treatment",
              "Treated is a small plus for a buyer with other pets at home (raise once) and a keep for the rest. Never "
              "treated is a new cost (a flea treatment at the vet): a buyer with pets at home or on a tight budget "
              "lowers once; others keep and say they will treat."),
    "microchip": ("whether the cat is microchipped",
                  "Microchipped is a plus for a buyer near a busy road or who wants the cat outdoors (raise once); "
                  "others keep. Not microchipped is a small new cost (a vet visit): a buyer on a tight budget lowers "
                  "once; others keep and say they will have it done."),
    "litter": ("whether the cat is litter trained",
               "Fully litter trained is what buyers expect: they keep. Still learning, or kicking litter everywhere, is "
               "a small daily effort: a buyer who wants a neat home or is out long hours lowers once; others keep."),
    "coat": ("how long the cat's coat is or how much brushing it needs",
             "Coat length is a personal taste: a buyer whose card loves a fluffy cat raises once for a long coat; one "
             "whose card wants little brushing, or who is allergic, lowers once for it; everyone else keeps."),
}
LIFE_KNOWLEDGE = ("- A cat's food is changed slowly, over about a week, mixing the old food with the new, to avoid an "
                  "upset stomach.\n- A vet puts in a microchip in a quick visit; it lets a lost cat be returned home.\n"
                  "- Fleas and ear mites are treated with products from the vet; untreated fleas spread to the other "
                  "pets in a home.")


def life(seed: int) -> str:
    """The life-facts section (owner, 2026-10-14; stories.LIFE_FACTS): this cat's one or two new facts, the question
    each answers, how buyers take them and the general facts behind that."""
    c = S.sale_for_seed(seed)["cat"]
    rows = "\n".join(f"- {fact} (a buyer asks {LIFE_ASK[k][0]})" for k, fact in zip(c["life_kinds"], c["life"],
                                                                                    strict=True))
    take = "\n".join(f"- {LIFE_ASK[k][1]}" for k in dict.fromkeys(c["life_kinds"]))
    return f'''

## Also on this cat's sheet: what real adopters ask (added 2026-10-14; checked by check_life.py)

Real adopters ask a shelter or seller things no sheet had before. This cat's sheet has these true facts too:
{rows}
For each, one buyer whose card makes the question natural asks it in their own words, and the seller answers
truly from the sheet (in their own style). Do not add any other new fact. How buyers take the answers:
{take}
General cat facts for these (buyers may use them):
{LIFE_KNOWLEDGE}
'''


def append_life(folder: Path, first: int, last: int) -> None:
    """Add the life section to prompts already written in folder (their other text is left as it is), and write
    life_facts.json for check_life.py."""
    facts = {}
    for seed in range(first, last + 1):
        c = S.sale_for_seed(seed)["cat"]
        if "life" not in c:
            continue
        path = folder / f"seed_{seed}.md"
        text = path.read_text()
        cut = text.find("\n\n## Also on this cat's sheet: what real adopters ask")
        path.write_text((text if cut < 0 else text[:cut]).rstrip("\n") + "\n" + life(seed))
        facts[str(seed)] = {"kinds": c["life_kinds"], "facts": c["life"]}
    (folder / "life_facts.json").write_text(json.dumps(facts, indent=1))
    print(f"life facts for {len(facts)} prompts in {folder}")


def main() -> None:
    """Write one prompt per seed (with --quotas: the lessons and this seed's scene list too)."""
    first, last, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    if "--append-life" in sys.argv:
        append_life(out, first, last)
        return
    seeds = list(range(first, last + 1))
    out.mkdir(parents=True, exist_ok=True)
    lists, overused = quota_lists(seeds) if "--quotas" in sys.argv else ({}, [])
    for seed in seeds:
        extra = lessons(lists[seed], overused) if lists else ""
        if "--human" in sys.argv:
            extra += human(seed)
        if "life" in S.sale_for_seed(seed)["cat"]:
            extra += life(seed)
        (out / f"seed_{seed}.md").write_text(prompt(seed, extra))
    if lists:
        (out / "scene_lists.json").write_text(json.dumps({str(s): v for s, v in lists.items()}, indent=1))
    print(f"{len(seeds)} prompts in {out}")


if __name__ == "__main__":
    main()
