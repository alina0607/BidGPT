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

The small-words batch (data/written_sales/prompts_3000_plan.md, seeds stories.PAIRED_SEEDS) adds, per seed, which
answers use which small word, a control and a "but" answer (cue_plan.json, checked by the folder's check_cues.py),
and writes the folder's checker copies as prompts_2900_2999 has them:
    PYTHONPATH=. python tools/external_prompt.py 3000 3049 ../data/written_sales/prompts_3000_3049 --quotas \
        --quota-target 50 --cues --checkers-from ../data/written_sales/prompts_2900_2999

Each prompt holds that seed's cat, seller style and ten buyers (stories.seed_sale), the general cat facts for the
topics they touch, every rule the training sales are held to, and the output format tools/write_sales.py reads. The
answer for seed N is saved as <specs_dir>/N.json and checked with `python tools/write_sales.py <specs_dir> <out>`.
Rules added after reading Muse's first sale (devlog 2026-10-12 §7) are marked as such in the text.
"""

from __future__ import annotations

import json
import random
import re
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


# The small words (data/written_sales/prompts_3000_plan.md, approved by the owner 2026-10-29; seeds stories.PAIRED_SEEDS):
# every paired seed's prompt says which target answers to give with which small word, one control answer (a negation
# that does not turn the direction), one answer joining a good and a bad fact with "but" or "though", and the held-out
# words no line may use. cue_plan.json keeps all of it for check_cues.py.
HELD_OUT_WORDS = ("barely", "hardly", "ain't", "no longer", "rarely", "seldom", "even though", "there's no denying")
HELD_OUT_PATTERN = r"\b(barely|hardly|rarely|seldom|no longer|even though)\b|\bain'?t\b|\bthere'?s no denying\b"
CUE_FAMILIES = {   # family: (the words of it check_cues.py looks for, how the prompt names it)
    "nt": (r"n't\b|\b(dont|doesnt|isnt|hasnt|wont|cant|didnt|wasnt|havent|couldnt|wouldnt|arent)\b",
           "n't (doesn't, isn't, hasn't, can't, won't; typed as doesnt or isnt is fine)"),
    "not": (r"\bnot\b", "not (or not yet)"),
    "no": (r"\b(no|none|nothing)\b", "no, none or nothing"),
    "never": (r"\bnever\b", "never"),
    "without": (r"\bwithout\b", "without"),
    "little": (r"\b(very little|not much|only a little|little)\b", "very little, not much or only a little"),
    "frag": (r"^\W*(not really|nope|nah|no idea|dunno|idk|not sure|no\b)",
             "a player's fragment at the very start (Not really, Nope, Nah, No idea, Dunno), then the fact"),
    "intact": (r"\b(intact|entire)\b", "the word intact"), "entire": (r"\b(entire|intact)\b", "the word entire"),
    "positive": (r"\bpositive\b", "the word positive"), "negative": (r"\bnegative\b", "the word negative"),
    "months": (r"\b(months?|mos?|mths?)\b", "the number 2 and the word months"),
    "years": (r"\b(years?|yrs?)\b", "the number 2 and the word years"),
    "fussy": (r"\b(fussy|picky)\b", "the word fussy (or picky)"), "any brand": (r"\bany\b", "the words any brand"),
    "plain": ("", "no small word: say the fact plainly"),
}
FORMS = {"full": "a full sentence", "fragment": "a player's fragment",
         "typed": "typed the way this seller types (lowercase, short forms)"}
NEGATION_FAMILIES = ("nt", "not", "no", "never", "without", "little", "frag")
CUE_OPTIONS = {   # the families the targets of one value take in turn, by seed (plan 3.2 shares, 4.3 balance)
    "sheds a lot": ("plain", "never", "nt", "no", "plain"), "sheds very little": ("nt", "little", "frag", "little", "little"),
    ("shed", "U"): ("nt", "frag"),
    "spayed/neutered": ("plain", "nt", "no", "plain"),
    "not yet spayed/neutered": ("intact", "not", "entire", "frag", "intact", "nt", "entire", "intact"),
    "2 months": ("months",), "2 years": ("years",),
    "fully vaccinated, vet check last month": ("plain", "no"), "finished the kitten vaccine series": ("nt",),
    "no vaccination records": ("no", "without"), "no vaccines yet": ("not",),
    "recovering from a cold": ("not",), "has some stiff joints": ("frag",),
    "tested negative for FIV and FeLV": ("negative", "no", "negative"), "tested FIV positive": ("positive",),
    "never tested for FIV or FeLV": ("never", "nt"),
    "is fine with children": ("plain", "nt", "no"), "doesn't like children": ("nt", "not", "frag"),
    "has never met children": ("never", "without"),
    "has lived with a dog": ("not", "plain", "nt"), "doesn't get on with dogs": ("nt", "frag", "not"),
    "has never met a dog": ("frag", "never"),
    "has lived with another cat": ("not", "nt"), "doesn't get on with other cats": ("nt", "not"),
    "has never met another cat": ("never", "frag"),
    "independent": ("little", "not", "nt"), "very cuddly": ("plain", "nt"), "follows people around like a dog": ("never",),
    "uses a scratching post, never the sofa": ("never", "not", "nt"),
    "a bit feisty, scratches when annoyed": ("plain", "nt", "plain"),
    "indoor-only": ("never", "nt", "not"), "indoor-outdoor": ("plain", "nt"), "used to roaming a farm": ("plain",),
    "playful and noisy at night": ("not", "plain"), "meows loudly for breakfast": ("no", "never"),
    "bloodwork last month was normal": ("no", "no"),
    "bloodwork shows early kidney disease, on a kidney diet": ("plain",), "no bloodwork in years": ("nt",),
    "no papers": ("never", "without"), ("papers", "U"): ("nt", "frag"),
    "eats wet food twice a day, any brand": ("any brand", "not"),
    "a fussy eater, only eats chicken wet food": ("fussy",), "eats one brand of dry food and nothing else": ("no",),
    "treated for fleas and ear mites last month": ("no",), "on a monthly flea treatment": ("without",),
    "has never had a flea treatment": ("never", "nt"),
    "fully litter trained": ("plain", "no"), "still learning to use the litter box": ("not",),
    "litter trained but kicks litter everywhere": ("plain",),
    "loves being picked up and carried": ("nt", "nt"), "hates being picked up but sits close by": ("not", "plain"),
    "microchipped": ("plain", "no"), "not microchipped": ("not", "frag"),
    "found as a stray and never claimed": ("never", "never"),
}
CUE_OPTIONS_HEARD = {   # 3050 on (more targets per value): keep one word under 40% of a topic (plan 3.2)
    "loves being picked up and carried": ("nt", "no", "plain"), "bloodwork last month was normal": ("no", "nt", "plain"),
    "found as a stray and never claimed": ("never", "no"),
}
GOOD_NEWS = {   # (topic, side): good for most buyers ("+"), bad ("-"), or neither (plan 4.3 counts + and -)
    ("shed", "A"): "-", ("shed", "B"): "+", ("spay", "A"): "+", ("spay", "B"): "-", ("vacc", "A"): "+",
    ("vacc", "B"): "-", ("fiv", "A"): "+", ("fiv", "B"): "-", ("kids", "A"): "+", ("kids", "B"): "-",
    ("dogs", "A"): "+", ("dogs", "B"): "-", ("cats", "A"): "+", ("cats", "B"): "-", ("scratch", "A"): "+",
    ("scratch", "B"): "-", ("noise", "B"): "-", ("senior", "A"): "+", ("senior", "B"): "-", ("papers", "B"): "-",
    ("food", "A"): "+", ("food", "B"): "-", ("pests", "A"): "+", ("pests", "B"): "-", ("litter", "A"): "+",
    ("litter", "B"): "-", ("handling", "A"): "+", ("handling", "B"): "-", ("microchip", "A"): "+",
    ("microchip", "B"): "-",
}
TOPIC_WORDS = {   # how check_cues.py finds the answer about a topic
    "shed": r"shed|fur\b|hair", "spay": r"spay|neuter|fix|intact|entire|\bop\b|operation|snip",
    "age": r"month|\bmos?\b|year|\byrs?\b|\bold\b|\bage\b", "vacc": r"shot|vacc|jab|booster",
    "fiv": r"fiv|felv|test", "kids": r"kid|child", "dogs": r"dog", "cats": r"\bcats?\b|kitt|feline",
    "clingy": r"independent|cuddl|follow|clingy|company|alone|lap\b|side\b|own thing|attention",
    "scratch": r"scratch|sofa|couch|post\b|claw", "home": r"indoor|inside|outside|outdoor|\bout\b|roam|farm|garden",
    "noise": r"meow|noisy|noise|loud|night|breakfast|yell|quiet|sleep", "senior": r"blood|kidney|test",
    "papers": r"paper|pedigree|registered|purebred|pure breed", "food": r"\beat|food|chicken|brand|wet|dry|fussy|picky",
    "pests": r"flea|mite", "litter": r"litter|box", "handling": r"pick|held|hold|carr|lap\b",
    "microchip": r"chip", "origin": r"stray|found|claim",
}
SIDE_WORDS = {   # (topic, side): words the answer must have to be on that side, and words it must not have
    ("shed", "A"): (r"a lot|lots|loads|tons|everywhere|heaps|constant|non ?stop|stops?\b|end to|plenty|so much",
                    r"\bvery little\b|\bnot much\b|n't shed much|\blittle\b"),
    ("shed", "B"): (r"little|much|\bbit\b|\bno (real )?(shed|fur)|n't shed\b|\bnot shed", r"a lot|loads|tons|everywhere"),
    ("shed", "U"): (r"know|sure|idea|tell|dunno|idk|\bsay\b", None),
    ("spay", "A"): (r"spay|neuter|fixed|done", r"\b(not yet|intact|entire|nope|nah)\b|n't (been )?(spay|neuter|fix|done)"),
    ("spay", "B"): (r"\b(not|yet|intact|entire|nope|nah|no)\b|n't", None),
    ("age", "A"): (r"\b(2|two)\b", r"\b(years?|yrs?)\b"), ("age", "B"): (r"\b(2|two)\b", r"\b(months?|mos?)\b"),
    ("vacc", "A"): (r"done|all|finished|full|up to date|every|complete|no more|n't need",
                    r"no (shots|vacc|records|jabs)|not (vaccinated|had)|never (had|been)|hasn't had"),
    ("vacc", "B"): (r"\b(no|none|not|never|without)\b|n't", None),
    ("vacc", "U"): (r"know|sure|idea|dunno|idk|\bsay\b|tell", None),
    ("fiv", "A"): (r"negative|\bno fiv|n't have|free", r"\bpositive\b"), ("fiv", "B"): (r"positive", r"\bnegative\b"),
    ("fiv", "U"): (r"never|not tested|n't been tested|untested|no test", r"\b(positive|negative)\b"),
    ("kids", "A"): (r"fine|great|good|love|mind|trouble|problem|ok", r"n't like|not good|aren't her|never met"),
    ("kids", "B"): (r"n't like|not good|n't get on|n't her|not a fan|not (her|his) thing|n't (his|her) thing|dislike|hate",
                    None),
    ("kids", "U"): (r"never|n't met|without|not been around|n't been around", None),
    ("dogs", "A"): (r"lived|fine|used to|not new|mind", r"n't get on|never met"),
    ("dogs", "B"): (r"n't get on|not good|n't (his|her) thing|not (his|her) thing|aren't for|n't like|n't do", None),
    ("dogs", "U"): (r"never|n't met", None),
    ("cats", "A"): (r"lived|fine|used to|not new|mind", r"n't get on|never met"),
    ("cats", "B"): (r"n't get on|not good|n't (his|her) thing|not (his|her) thing|n't like|n't do", None),
    ("cats", "U"): (r"never|n't met", None),
    ("clingy", "A"): (r"independent|own thing|not clingy|n't need|on (her|his) own", None),
    ("clingy", "B"): (r"cuddl|follow|clingy|side|lap|n't get enough|never leaves", None),
    ("scratch", "A"): (r"post", None), ("scratch", "B"): (r"scratch|feisty|annoy", r"\bpost\b"),
    ("home", "A"): (r"indoor|inside|never goes out|n't go out|not an outdoor", r"out all day|roam"),
    ("home", "B"): (r"outdoor|outside|out\b|roam|farm", r"never goes out|n't go out|indoor only|indoor-only"),
    ("noise", "B"): (r"meow|noisy|noise|loud|yell|quiet|night|breakfast", None),
    ("senior", "A"): (r"normal|fine|nothing wrong|no problem|clear|good", r"kidney"),
    ("senior", "B"): (r"kidney|no bloodwork|n't had|not (been )?tested|years", r"\bnormal\b"),
    ("papers", "B"): (r"paper", None), ("papers", "U"): (r"know|sure|idea|dunno|idk|\bsay\b", None),
    ("food", "A"): (r"any|not fussy|n't fussy|n't picky|whatever", r"only|nothing else"),
    ("food", "B"): (r"only|fussy|picky|nothing else|one brand|just", None),
    ("pests", "A"): (r"treat|dose|monthly|last month|drops", r"never had|no flea treatment"),
    ("pests", "B"): (r"never|no\b|n't had|n't been|not been", None),
    ("litter", "A"): (r"trained|no training|uses", r"learning|kick"), ("litter", "B"): (r"learning|kick|not fully|n't", None),
    ("handling", "A"): (r"love|n't mind|happy|enjoy", r"hate|n't like"),
    ("handling", "B"): (r"hate|n't like|not a fan|won't|n't want", None),
    ("microchip", "A"): (r"chip", r"\bnot\b|\bno chip|n't|nope"), ("microchip", "B"): (r"\bnot\b|\bno\b|n't|nope", None),
    ("origin", "C"): (r"stray", None),
}
TAKE = {   # (topic, side): how the buyers the target turns take it (plan 3.1, last column)
    ("shed", "A"): "the allergic buyer, and any buyer whose card minds shedding, lowers; everyone else keeps",
    ("shed", "B"): "the allergic buyer, and any buyer whose card minds shedding, raises; everyone else keeps",
    ("shed", "U"): "nobody moves: the allergic buyer keeps and says they will spend time with this cat first to see",
    ("spay", "A"): "a breeder withdraws (she cannot be bred); everyone else keeps",
    ("spay", "B"): "a breeder raises; everyone else lowers once for the spay or neuter bill; an allergic buyer hearing a "
                   "kitten under puberty keeps (it can be done before then)",
    ("age", "A"): "only a buyer whose card asks about age itself moves (a kitten to raise: raise; a settled grown cat: "
                  "lower); others keep. A kitten's buyers name kitten things only when the sheet says them "
                  "(shots still to finish, a spay still to do)",
    ("age", "B"): "only a buyer whose card asks about age itself moves (a settled grown cat: raise; a kitten to raise: "
                  "lower); others keep. Only now may a buyer speak of a young adult or a yearly checkup",
    ("vacc", "A"): "the buyer who wants things in excellent shape raises; everyone else keeps",
    ("vacc", "B"): "every buyer lowers once for the shots (a bill)",
    ("vacc", "U"): "nobody moves: each buyer keeps and says how they will find out (the vet's records, a vet check)",
    ("fiv", "A"): "the buyer with an older cat at home raises; everyone else keeps (an adult)",
    ("fiv", "B"): "the buyer with an older cat at home lowers (or the seller refuses that buyer); everyone else lowers "
                  "once for the vet care",
    ("fiv", "U"): "the buyer with an older cat at home lowers once (a test is needed first); everyone else keeps",
    ("kids", "A"): "the buyer with young kids raises; everyone else keeps",
    ("kids", "B"): "the buyer with young kids lowers or withdraws; everyone else keeps",
    ("kids", "U"): "the buyer with young kids lowers once (their card is about this); everyone else keeps",
    ("dogs", "A"): "the buyer with two dogs raises; everyone else keeps",
    ("dogs", "B"): "the buyer with two dogs lowers or withdraws; everyone else keeps",
    ("dogs", "U"): "the buyer with two dogs lowers once (their card is about this); everyone else keeps",
    ("cats", "A"): "the buyer with an older cat at home raises; everyone else keeps",
    ("cats", "B"): "the buyer with an older cat at home lowers or withdraws; everyone else keeps",
    ("cats", "U"): "the buyer with an older cat at home lowers once (their card is about this); everyone else keeps",
    ("clingy", "A"): "the buyer who works long hours raises; a buyer whose card wants a lap cat keeps",
    ("clingy", "B"): "the buyer who works long hours lowers; a buyer whose card wants a lap cat raises",
    ("scratch", "A"): "the buyer who cares how things look raises; everyone else keeps",
    ("scratch", "B"): "the buyer with young kids lowers; everyone else keeps",
    ("home", "A"): "the buyer on a busy road (and one in a small flat) raises; the hiker lowers",
    ("home", "B"): "the buyer on a busy road (and one in a small flat) lowers; the hiker raises",
    ("noise", "B"): "every buyer who sleeps at night lowers once (lost sleep); the night shift nurse keeps",
    ("senior", "A"): "most buyers keep",
    ("senior", "B"): "every buyer lowers once (the vet bills or the tests it needs)",
    ("papers", "B"): "a breeder withdraws (no registration); a buyer who prizes pedigree lowers once; everyone else keeps",
    ("papers", "U"): "nobody moves: each buyer keeps and says how they will find out (ask for the breeder's papers)",
}
TAKE_HEARD = {   # 3050 on: sides TAKE leaves to the life-fact rule, said in the target's own row
    ("litter", "A"): "nobody moves: fully litter trained is what every buyer expects, so they keep",
}
FLIP_SIGNS = {   # (topic, side): {persona: the sign plan 3.1 asks for}; check_cues.py lists the other signs to read
    ("shed", "A"): {"is allergic and worried about fur": "-"}, ("shed", "B"): {"is allergic and worried about fur": "+"},
    ("shed", "U"): {"is allergic and worried about fur": "0"},
    ("spay", "A"): {"breeds show cats": "out"}, ("spay", "B"): {"breeds show cats": "+"},
    ("vacc", "A"): {"wants things in excellent shape": "+"}, ("vacc", "B"): {"wants things in excellent shape": "-"},
    ("vacc", "U"): {"wants things in excellent shape": "0"},
    ("fiv", "A"): {"already has an older cat at home": "+"}, ("fiv", "B"): {"already has an older cat at home": "-"},
    ("fiv", "U"): {"already has an older cat at home": "-"},
    ("kids", "A"): {"is buying for a family with young kids": "+"},
    ("kids", "B"): {"is buying for a family with young kids": "-"},
    ("kids", "U"): {"is buying for a family with young kids": "-"},
    ("dogs", "A"): {"already has two dogs": "+"}, ("dogs", "B"): {"already has two dogs": "-"},
    ("dogs", "U"): {"already has two dogs": "-"},
    ("cats", "A"): {"already has an older cat at home": "+"}, ("cats", "B"): {"already has an older cat at home": "-"},
    ("cats", "U"): {"already has an older cat at home": "-"},
    ("clingy", "A"): {"works long hours and has little free time": "+"},
    ("clingy", "B"): {"works long hours and has little free time": "-"},
    ("scratch", "A"): {"cares a lot about how things look": "+"},
    ("scratch", "B"): {"is buying for a family with young kids": "-"},
    ("home", "A"): {"lives on a busy road": "+", "spends weekends hiking and wants an adventure cat": "-"},
    ("home", "B"): {"lives on a busy road": "-", "spends weekends hiking and wants an adventure cat": "+"},
    ("noise", "B"): {"is a nurse working night shifts": "0"},
    ("papers", "B"): {"breeds show cats": "out", "loves rare, premium and pedigree things": "-"},
    ("papers", "U"): {"breeds show cats": "0", "loves rare, premium and pedigree things": "0"},
    ("pests", "A"): {"already has two dogs": "+"}, ("pests", "B"): {"already has two dogs": "-"},
    ("handling", "A"): {"is buying for a family with young kids": "+"},
    ("handling", "B"): {"is buying for a family with young kids": "-"},
    ("microchip", "A"): {"lives on a busy road": "+"}, ("microchip", "B"): {"is a student on a tight budget": "-"},
}
CONTROLS = {   # kind: (how many in the batch, what to write, an example about another cat)
    "unrelated": (15, "an answer with a negation in a clause about something not on the sheet, then the fact",
                  "No clue what her first name was, but she's microchipped."),
    "double_neg": (12, "an answer with a double negation, or a negation that says the fact the positive way",
                   ("Not a morning goes by without her yelling for breakfast. / She's never once used the sofa, only "
                    "her post.")),
    "not_only": (10, "an answer joining two good facts with not only ... but (also)", ("Not only is she spayed, she "
                 "tested negative as well.")),
    "no_doubt": (8, "an answer that starts with no doubt, then the fact", "No doubt about it, she sheds a lot."),
    "neg_question": (5, ("a buyer's question with the fact turned negative (\"She's not ..., is she?\") and the "
                         "seller's answer that it is true (\"She is, ...\")"),
                     "Q: She's not spayed, is she? A: She is, done last spring."),
}
CONTROL_WORDS = {   # kind: what check_cues.py looks for in the answer (neg_question: in the question)
    "unrelated": r"(n't|\bnot\b|\bno\b|\bnever\b)[^.?!]*\b(but|though)\b",
    "double_neg": r"\b(not|never|no)\b[^.?!]*(\bwithout\b|\bonly\b|n't)|n't[^.?!]*\bwithout\b",
    "not_only": r"\bnot only\b[^.?!]*\bbut\b", "no_doubt": r"\bno doubt\b",
    "neg_question": r"(n't|\bnot\b)[^?]*\?",
}
YES = r"^\W*(yes|yeah|yep|yea|yup|she is|he is|she has|he has|she does|he does|it is|of course|sure|actually)"
GOOD_FACTS = {"spayed/neutered", "finished the kitten vaccine series", "fully vaccinated, vet check last month",
              "microchipped and healthy", "bloodwork last month was normal", "tested negative for FIV and FeLV",
              "has lived with a dog", "has lived with another cat", "is fine with children", "indoor-only",
              "indoor with a catio", "calm and lazy", "very cuddly",
              "uses a scratching post, never the sofa", S.SHEDS_LITTLE, "microchipped", "fully litter trained",
              "treated for fleas and ear mites last month", "on a monthly flea treatment",
              "loves being picked up and carried", "eats wet food twice a day, any brand"}
BAD_FACTS = {"not yet spayed/neutered", "no vaccines yet", "dewormed but not vaccinated", "no vaccination records",
             "had the first of three kitten vaccines", "had two of three kitten vaccines", S.NO_VET,
             "recovering from a cold", "bloodwork shows early kidney disease, on a kidney diet", "no bloodwork in years",
             "on medication for an overactive thyroid", "has some stiff joints", "tested FIV positive",
             *S.AGAINST.values(), "a bit feisty, scratches when annoyed", "playful and noisy at night", "sheds a lot",
             "meows loudly for breakfast", "only drinks from a running tap", "picky about litter",
             "has never had a flea treatment", "not microchipped", "still learning to use the litter box",
             "litter trained but kicks litter everywhere", "a fussy eater, only eats chicken wet food",
             "eats one brand of dry food and nothing else", "hates being picked up but sits close by"}
YES_FACTS = ("spayed/neutered", "fully vaccinated, vet check last month", "finished the kitten vaccine series",
             "tested negative for FIV and FeLV", "has lived with a dog", "has lived with another cat",
             "is fine with children", "indoor-only", "microchipped", "fully litter trained")
HABITS = ("meows loudly for breakfast", "uses a scratching post, never the sofa", "only drinks from a running tap",
          "afraid of the vacuum", "picky about litter", "sheds a lot", *S.TEMPER, *S.LIFE_FACTS["handling"][0],
          *S.LIFE_FACTS["food"][0], "on a monthly flea treatment")
FACT_WORDS = (   # (a piece of a sheet fact, the words that show it in an answer); the first piece found counts
    ("spayed", r"spay|neuter|fix|intact|entire"), ("vaccin", r"shot|vacc|jab|booster"), ("bloodwork", r"blood|kidney"),
    ("FIV", r"fiv|felv|test"), ("microchip", r"chip"), ("cold", r"cold|sneez|sniffl"), ("thyroid", r"thyroid|pill"),
    ("joints", r"joint|stiff"), ("vet visit", r"\bvet\b"), ("dewormed", r"worm"), ("cuddly", r"cuddl"),
    ("independent", r"independent|own thing"), ("shy", r"shy|stranger"), ("noisy at night", r"night|noisy|play"),
    ("calm and lazy", r"calm|lazy"), ("feisty", r"scratch|feisty|annoy"), ("follows", r"follow"),
    ("running tap", r"\btap\b"), ("scratching post", r"post|sofa"), ("sheds", r"shed|fur"),
    ("breakfast", r"breakfast|meow"), ("picky about litter", r"litter"), ("vacuum", r"vacuum|hoover"),
    ("indoor-only", r"indoor|inside|out\b"), ("catio", r"catio"), ("indoor-outdoor", r"outdoor|outside|out\b"),
    ("farm", r"farm|roam"), ("stray", r"stray|found"), ("moved abroad", r"moved|abroad"), ("allergic", r"allerg"),
    ("care home", r"care home"), ("unplanned litter", r"litter|neighbou?r"), ("food", r"food|eat|chicken|brand|wet|dry"),
    ("flea", r"flea|mite"), ("litter", r"litter"), ("picked up", r"pick|held|hold|carr"), ("hold", r"pick|held|hold"),
    ("papers", r"paper|pedigree|purebred"), ("children", r"kid|child"), ("dog", r"dog"), ("cat", r"\bcats?\b"),
)


def fact_words(fact: str) -> str:
    """The words that show a sheet fact in an answer (check_cues.py looks for them)."""
    return next(words for piece, words in FACT_WORDS if piece in fact)


def sheet_facts(c: dict) -> list[str]:
    """Every fact on a cat's sheet that can be said in one clause (the age and breed aside)."""
    return [c["home"], c["fixed"], c["health"], c["virus"], *c["company"], c["temperament"], *c["quirks"],
            *c.get("extra", []), *c.get("life", [])]


def cue_plan(seeds: list[int]) -> dict:
    """Each paired seed's targets with their small word and form, its control and its "but" answer (plan 2.1 and 7):
    target families go round each value's CUE_OPTIONS in seed order; control kinds are handed out by the most left of
    CONTROLS' quotas that the sheet allows, facts drawn from the sheet's other facts."""
    turn: dict[object, int] = {}
    left = {k: v[0] for k, v in CONTROLS.items()}
    members: dict[str, list[int]] = {}
    for seed in sorted(S.PAIRED_SEEDS):
        for t in S.PAIRED_SEEDS[seed]:
            members.setdefault(t[3], []).append(seed)
    plan = {}
    for seed in seeds:
        s = S.sale_for_seed(seed)
        c, rng = s["cat"], random.Random(seed * 7 + 3)
        targets = []
        persona_name = {x["persona"]: x["name"] for x in s["buyers"]}
        for entry in S.PAIRED_SEEDS[seed]:
            topic, side, value, group = entry[:4]
            key = value if value in CUE_OPTIONS else (topic, side)
            options = CUE_OPTIONS_HEARD.get(key, CUE_OPTIONS[key]) if seed >= S.HEARD_FROM else CUE_OPTIONS[key]
            family = options[turn.get(key, 0) % len(options)]
            turn[key] = turn.get(key, 0) + 1
            words, not_words = SIDE_WORDS[(topic, side)]
            targets.append({
                "topic": topic, "side": side, "value": value, "group": group, "cue": family,
                "cue_words": CUE_FAMILIES[family][0],
                "form": "fragment" if family == "frag" else "typed" if "typos" in s["seller_style"] else "full",
                "news": GOOD_NEWS.get((topic, side), "0"),
                "flip": [p for p in S.paired_buyers(seed) if p in S.PAIRED_BUYERS.get(topic, ())
                         + S.PAIRED_GROUP_BUYERS.get(group, ())],
                "expect": FLIP_SIGNS.get((topic, side), {}), "topic_words": TOPIC_WORDS[topic],
                "side_words": words, "not_words": not_words})
            if len(entry) > 4:              # 3050 on: how the buyers it turns hear it, and who asks
                if (topic, side) in TAKE_HEARD:         # a side that turns nobody (found writing 3053)
                    targets[-1]["flip"] = []
                asker = S.BUYER[entry[5]] if entry[5] else None
                targets[-1].update({"heard": entry[4], "asker": asker, "asker_name": persona_name.get(asker)})
        if seed in S.PAIRED_BUTS:
            plan[str(seed)] = _heard_entry(seed, s, targets, members)
            continue
        said = {t[2] for t in S.PAIRED_SEEDS[seed]}
        free = [f for f in sheet_facts(c) if f not in said]
        kind = max((k for k in CONTROLS if left[k] and _control_fits(k, free)),      # the kind least used so far
                   key=lambda k: left[k] / CONTROLS[k][0])
        left[kind] -= 1
        control = _control(kind, free, rng)
        rest = [f for f in free if f not in control["facts"]] or free
        good = (sorted(f for f in rest if f in GOOD_FACTS) or sorted(f for f in sheet_facts(c) if f in GOOD_FACTS)
                or sorted(f for f in rest if f not in BAD_FACTS))      # no good fact at all: a neutral one
        bad = sorted(f for f in rest if f in BAD_FACTS) or sorted(f for f in sheet_facts(c) if f in BAD_FACTS)
        pair = [rng.choice(good), rng.choice(bad)]
        order = "good but bad" if seed % 2 == 0 else "bad but good"
        facts = pair if order == "good but bad" else pair[::-1]
        plan[str(seed)] = {
            "targets": targets, "control": control,
            "contrast": {"order": order, "facts": facts, "fact_words": [fact_words(f) for f in facts]},
            "pair_with": {t["group"]: [x for x in members[t["group"]] if x != seed] for t in targets},
            "buyers": S.paired_buyers(seed), "names": {x["name"]: x["persona"] for x in s["buyers"]}, "fragments": 2}
    return {"held_out": list(HELD_OUT_WORDS), "held_out_pattern": HELD_OUT_PATTERN,
            "families": {k: v[0] for k, v in CUE_FAMILIES.items()}, "control_words": CONTROL_WORDS, "yes": YES,
            "fragment": CUE_FAMILIES["frag"][0], "stop": sorted(S.STOP), "allowed": _sheet_phrases(), "seeds": plan}


def _heard_entry(seed: int, s: dict, targets: list[dict], members: dict[str, list[int]]) -> dict:
    """A 3050-on seed's cue plan: its targets, and the control and two "but" answers stories.py froze for it."""
    names = {x["persona"]: x["name"] for x in s["buyers"]}
    kind, facts = S.PAIRED_CONTROLS[seed]
    control = {"kind": kind, "facts": list(facts), "fact_words": [fact_words(f) for f in facts],
               "words": CONTROL_WORDS[kind]}
    buts = []
    for but_kind, connector, heard, asker, up, down, but_facts in S.PAIRED_BUTS[seed]:
        who = {k: S.BUYER[v] if v else None for k, v in (("asker", asker), ("up", up), ("down", down))}
        buts.append({"kind": but_kind, "connector": connector, "heard": heard, "facts": list(but_facts),
                     "fact_words": [fact_words(f) for f in but_facts],
                     **who, **{f"{k}_name": names.get(v) for k, v in who.items()}})
    return {"targets": targets, "control": control, "buts": buts,
            "pair_with": {t["group"]: [x for x in members[t["group"]] if x != seed] for t in targets},
            "buyers": S.paired_buyers(seed), "names": {x["name"]: x["persona"] for x in s["buyers"]}, "fragments": 2}


def _sheet_phrases() -> list[str]:
    """Every fact a sheet can hold and every persona: a line may share these with a test sentence."""
    lists = (S.HOME, S.HEALTH_KITTEN, S.HEALTH_ADULT, S.HEALTH_SENIOR, S.VIRUS, *S.COMPANY.values(), S.AGAINST.values(),
             S.TEMPER, S.QUIRKS, (S.SHEDS_LITTLE, "not yet spayed/neutered", "spayed/neutered"), S.PERSONAS,
             *(x for kind in S.LIFE_FACTS.values() for x in kind))
    return sorted({x for xs in lists for x in xs})


def _control_fits(kind: str, free: list[str]) -> bool:
    """Whether the sheet's free facts allow a control of this kind."""
    if kind == "not_only":
        return sum(f in GOOD_FACTS for f in free) >= 2
    if kind == "neg_question":
        return any(f in YES_FACTS for f in free)
    return bool(free)


def _control(kind: str, free: list[str], rng: random.Random) -> dict:
    """The control answer of one kind, on facts drawn from the sheet's free facts."""
    if kind == "not_only":
        facts = rng.sample(sorted(f for f in free if f in GOOD_FACTS), 2)
    elif kind == "neg_question":
        facts = [rng.choice(sorted(f for f in free if f in YES_FACTS))]
    elif kind == "double_neg" and any(f in HABITS for f in free):
        facts = [rng.choice(sorted(f for f in free if f in HABITS))]
    else:                                   # an unknown ("has never met a dog") only when nothing else is left
        known = [f for f in free if "never met" not in f and "never tested" not in f]
        facts = [rng.choice(sorted(known or free))]
    return {"kind": kind, "facts": facts, "fact_words": [fact_words(f) for f in facts],
            "words": CONTROL_WORDS[kind]}


def _said(t: dict) -> str:
    """A target's fact as the prompt names it."""
    if t["value"] is None:
        return {"shed": "how much the cat sheds", "papers": "whether the cat has papers"}[t["topic"]] + \
            " (not on the sheet: the seller does not know)"
    if t["topic"] == "age":
        return f"the age, {t['value']}"
    if t["topic"] == "vacc" and t["side"] == "U":
        return "the cat's vaccines (not on the sheet: the seller is not sure)"
    return t["value"]


def _heard(t: dict) -> str:
    """Who asks a 3050-on target, and how the buyers it turns hear it (plan section 2)."""
    if "heard" not in t:
        return ""
    if t["heard"] == "self":
        if t["asker"] and not t["flip"]:
            return f" Heard: asked by {t['asker_name']} ({t['asker']})."
        return (f" Heard: asked by {t['asker_name']} ({t['asker']}), a buyer it turns." if t["asker"]
                else " Heard: asked by a buyer it moves.")
    if t["heard"] == "other":
        return (f" Heard: asked by {t['asker_name']} ({t['asker']}), not by a buyer it turns: {t['asker_name']}'s "
                "card or greeting gives a reason to ask it (item 43). The buyers it turns still move as 'How "
                "buyers take it' says and say what they heard (item 42).")
    return (" Heard: said inside the answer to a question about something else: the seller first answers that "
            "question, then adds this fact as its own sentence, with the small word in that sentence's first six "
            "words. The buyers it turns still move as 'How buyers take it' says and say what they heard (item 42).")


BUT_KINDS = {   # plan 3.4: what each kind of "but" answer says, and an example about another cat (do not copy)
    "good but bad": ("the good fact first, then the bad one", '"Fully vaccinated, but no microchip."'),
    "bad but good": ("the bad fact first, then the good one", '"She\'s not spayed, but her shots are all done."'),
    "fragment": (("the first fact as a player's fragment at the very start (nah, nope, not really, no), then the "
                  "other one"), '"nah not with kids, but dogs fine" / "no papers tho, she\'s purebred"'),
    "caveat": (("this one fact, said with its own turn in the middle; the part after the connector decides the "
                "direction"), '"She sheds a bit, but only in spring."'),
}


def _but_row(k: int, b: dict) -> str:
    """One "but" answer of a 3050-on prompt: its facts, its connector, who asks and the buyers its halves move."""
    what, example = BUT_KINDS[b["kind"]]
    facts = " / ".join(b["facts"])
    asked = (f"asked by {b['asker_name']} ({b['asker']}), who is neither of the buyers below" if b["heard"] == "other"
             else f"asked by {b['asker_name']} ({b['asker']})")
    moved = (f"{b['up_name']} ({b['up']}) raises for the good half and {b['down_name']} ({b['down']}) lowers for the bad "
             "half, so a buyer who reads only one half gets it wrong" if b["up"]
             else f"{b['down_name']} ({b['down']}) lowers for the part after the connector")
    return (f"{k}. **{facts}** ({b['kind']}: {what}), joined with \"{b['connector']}\"; {asked}. {moved}; every "
            f"other buyer reacts to each half that concerns them. Example (another cat): {example}")


def cues(seed: int, entry: dict) -> str:
    """The section of one paired seed's prompt that says which small words its answers use (plan 2.2, 3, 4, 5.1)."""
    rows = []
    for k, t in enumerate(entry["targets"], 1):
        flip = f" Buyers it turns: the one who {'; the one who '.join(t['flip'])}." if t["flip"] else ""
        take = TAKE.get((t["topic"], t["side"])) or (TAKE_HEARD.get((t["topic"], t["side"])) if "heard" in t else None)
        front = (", within the first six words of the sentence that says it" if t.get("heard") == "inside"
                 else ", within the answer's first six words")
        word = CUE_FAMILIES[t["cue"]][1]
        if t.get("heard") == "inside":
            word = word.replace("at the very start", "at the start of its sentence")
        rows.append(f"{k}. **{_said(t)}**. Small word: {word}"
                    + ("" if t["cue"] == "plain" else front)
                    + f". Form: {FORMS[t['form']]}.{flip}" + (f" How buyers take it: {take}." if take else "")
                    + _heard(t))
    control = entry["control"]
    _, what, example = CONTROLS[control["kind"]]
    pairs = "; ".join(f"{g}: {', '.join(map(str, v))}" for g, v in entry["pair_with"].items())
    if "buts" in entry:
        return _heard_cues(entry, rows, what, example, pairs)
    contrast = entry["contrast"]
    return f'''

## This sale's small words (added 2026-10-29; checked by check_cues.py, read by the reviewer)

Why: the buyer model reads the topic of an answer but not its small words ("not", "never", "but"): "She sheds a lot"
and "She doesn't shed much" get the same reaction. This batch gives it answers that differ by those words and buyers
whose offer turns with them. Paired sales ({pairs}) carry the same topic with the opposite meaning.

**A. Target answers.** Each comes out in the answer to a buyer's question that truly asks it (usually a buyer it
turns), says exactly this side of the fact and nothing more (no detail the sheet does not give: not how she gets on
badly, not why), and puts the small word within the first six words: "Not microchipped, sorry", not "She came to me
from a farm and she's not microchipped" (older answers are cut from the end in the buyer's prompt, so a late small
word can be lost). The note keeps the same small word.
{chr(10).join(rows)}

**B. One control answer ({control["kind"]}): a negation that does NOT turn the direction.** Write {what}, about:
{"; ".join(control["facts"])}. Example (another cat; do not copy): "{example}" Every buyer reacts exactly as they would
to the plain fact: never lower because the sentence has "not", "no" or "never" in it.

**C. One answer joining two facts with "but" or "though"** ({contrast["order"]}): {contrast["facts"][0]}, then
{contrast["facts"][1]}. Each buyer reacts to both halves that concern them.

**D. Player fragments.** At least two answers start the way players type: "Not really, but ...", "nope, not yet",
"dunno tbh", "yea all shots done" (a fragment target above counts).

**E. Never use these words anywhere in the sale** (no answer, question, note, reaction, scene or decision; they are
kept out of training to test the model on them): {", ".join(HELD_OUT_WORDS)}.

**F. Buyers.** Every greeting ends with the kind of cat the buyer wants, following from the life it tells ("I sand
every boat hull I sell until it shines, and I'd like a cat kept with the same care", not "I'm a plumber, so I want a
healthy cat").
"So", "since" and "because" join only a real cause. A buyer with dogs or an older cat at home, hearing about
vaccines, FIV, fleas, dogs or other cats, says what it means for their own animal by the name from their greeting
(not a stock line like "Kitten shots are normal").
'''


def _heard_cues(entry: dict, rows: list[str], what: str, example: str, pairs: str) -> str:
    """The small-words section of a 3050-on prompt (data/written_sales/prompts_3050_plan.md): who hears each target,
    and two "but" answers whose halves move two buyers opposite ways."""
    control = entry["control"]
    buts = "\n".join(_but_row(k, b) for k, b in enumerate(entry["buts"], 1))
    return f'''

## This sale's small words and who hears them (added 2026-10-31; checked by check_cues.py, read by the reviewer)

Why: the buyer model reads the topic of an answer but not its small words ("not", "never", "but"), and when another
buyer asks, it leaves the buyer the answer is about on keep: "She doesn't like children" moved the parent when the
parent asked, and nobody when someone else asked. In the game nine buyers of ten hear every answer to someone else's
question. Paired sales ({pairs}) carry the same topic with the opposite meaning.

**A. Target answers.** Each says exactly this side of the fact and nothing more (no detail the sheet does not give:
not how she gets on badly, not why), and puts the small word within the first six words: "Not microchipped, sorry",
not "She came to me from a farm and she's not microchipped" (older answers are cut from the end in the buyer's prompt,
so a late small word can be lost). The note keeps the same small word. "Heard" says who asks: a buyer it turns hears
it the same way whoever asks, and moves as "How buyers take it" says, naming what they heard ("Hearing she doesn't like
children, ..."). Good news moves them as surely as bad news: a raise for good news is never left on keep (item 44).
{chr(10).join(rows)}

**B. One control answer ({control["kind"]}): a negation that does NOT turn the direction.** Write {what}, about:
{"; ".join(control["facts"])}. Example (another cat; do not copy): "{example}" Every buyer reacts exactly as they would
to the plain fact: never lower because the sentence has "not", "no" or "never" in it.

**C. Two answers that join two sides with "but", "though" or "tho"** (each in its own round, the connector as given).
Each says only sheet facts, and every buyer reacts to each half that concerns them (item 2):
{buts}

**D. Player fragments.** At least two answers start the way players type: "Not really, but ...", "nope, not yet",
"dunno tbh", "yea all shots done" (a fragment target or "but" answer above counts).

**E. Never use these words anywhere in the sale** (no answer, question, note, reaction, scene or decision; they are
kept out of training to test the model on them): {", ".join(HELD_OUT_WORDS)}.

**F. Buyers.** Every greeting ends with the kind of cat the buyer wants, following from the life it tells ("I sand
every boat hull I sell until it shines, and I'd like a cat kept with the same care", not "I'm a plumber, so I want a
healthy cat"). A buyer named above as asking something says in their greeting the life detail that makes them ask it.
"So", "since" and "because" join only a real cause. A buyer with dogs or an older cat at home, hearing about
vaccines, FIV, fleas, dogs or other cats, says what it means for their own animal by the name from their greeting
(not a stock line like "Kitten shots are normal").
'''


def write_cue_plan(folder: Path, seeds: list[int]) -> dict:
    """cue_plan.json for check_cues.py; returns the plan."""
    plan = cue_plan(seeds)
    (folder / "cue_plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    return plan


def write_checkers(folder: Path, seeds: list[int], overused: list[str], template: Path) -> None:
    """The checker copies a batch folder holds (as prompts_2900_2999 has them): check_sales.py from
    make_sale_checker, check_lessons.py from an earlier batch's copy with this batch's seeds, check_life.py and its
    life_facts.json."""
    from make_sale_checker import checker
    (folder / "check_sales.py").write_text(checker(seeds[0], seeds[-1]))
    lessons_src = (template / "check_lessons.py").read_text()
    indoor = {str(s) for s in seeds if S.sale_for_seed(s)["cat"]["home"] == "indoor-only"}
    for name, value in (("OVERUSED", set(overused)), ("INDOOR_ONLY", indoor), ("NO_OTHER_SEEDS", {str(s) for s in seeds})):
        lessons_src = re.sub(rf"^{name} = \{{.*?\}}$", f"{name} = {sorted(value)!r}".replace("[", "{").replace("]", "}"),
                              lessons_src, count=1, flags=re.MULTILINE)
    (folder / "check_lessons.py").write_text(lessons_src)
    (folder / "check_life.py").write_text((template / "check_life.py").read_text())
    life_facts = {str(s): {"kinds": c["life_kinds"], "facts": c["life"]}
                  for s in seeds for c in [S.sale_for_seed(s)["cat"]] if "life" in c}
    (folder / "life_facts.json").write_text(json.dumps(life_facts, indent=1))


def main() -> None:
    """Write one prompt per seed (with --quotas: the lessons and this seed's scene list too; --quota-target N: lists
    toward N scenes a thing, as seeds 2737 on used 50; --cues: the small-words section and cue_plan.json for the
    stories.PAIRED_SEEDS; --checkers-from FOLDER: the checker copies, check_lessons.py taken from FOLDER)."""
    first, last, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    if "--append-life" in sys.argv:
        append_life(out, first, last)
        return
    seeds = list(range(first, last + 1))
    out.mkdir(parents=True, exist_ok=True)
    target = int(sys.argv[sys.argv.index("--quota-target") + 1]) if "--quota-target" in sys.argv else 30
    lists, overused = quota_lists(seeds, target) if "--quotas" in sys.argv else ({}, [])
    plan = write_cue_plan(out, seeds) if "--cues" in sys.argv else None
    for seed in seeds:
        extra = lessons(lists[seed], overused) if lists else ""
        if "--human" in sys.argv:
            extra += human(seed)
        if "life" in S.sale_for_seed(seed)["cat"]:
            extra += life(seed)
        if plan:
            extra += cues(seed, plan["seeds"][str(seed)])
        (out / f"seed_{seed}.md").write_text(prompt(seed, extra))
    if lists:
        (out / "scene_lists.json").write_text(json.dumps({str(s): v for s, v in lists.items()}, indent=1))
    if "--checkers-from" in sys.argv:
        write_checkers(out, seeds, overused, Path(sys.argv[sys.argv.index("--checkers-from") + 1]))
    print(f"{len(seeds)} prompts in {out}")

if __name__ == "__main__":
    main()
