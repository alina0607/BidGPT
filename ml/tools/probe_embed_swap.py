"""Is a misread word a problem of the word's input vectors? Swap them and see what the buyer does (nothing is trained).

    cd ml && python tools/probe_embed_swap.py runs/sft_b3n/best.pt --cases cases.jsonl --out runs/sft_b3n/embed_swap.jsonl

One forward pass per condition, on the CPU, nothing sampled. The prompt ends with `<|react|>`; the first thing the
buyer writes is its sign ("+" raise, "0" hold, "-" cut, see stories.SIGNS), so the model's probability of each sign
at that place is the buyer's direction. A case names a prompt, the word to work on, the sign the answer should
give (`want`) and a familiar word that means the same (`donor`). The word's own pieces (as SentencePiece cuts them in
the prompt) get other input vectors through a hook on the embedding layer. The embedding table is not touched: it is
tied to the output layer, so editing a row would also change what the model says for that token.

    base          nothing changed
    text_swap     the donor word written into the prompt (the model reads it the normal way): the ceiling
    donor         every piece of the word gets the mean vector of the donor's pieces
    pieces_mean   every piece gets the mean vector of the word's own pieces
    context_mean  every piece gets the mean vector of the other tokens of the seller's answer
    unrelated     every piece gets the vector of an unrelated word (UNRELATED)
    random        every piece gets a random vector of the mean norm of a table row (mean of RANDOM_DRAWS draws)
    negated       every piece gets minus its own vector

A case is {"prompt", "word", "want", "donor"} or {"from": ["new", 12], "answer", "asked", "word", "want", "donor"}:
the second takes the prompt of the react example at that line of val_{set}_format*.jsonl in --sets and puts `answer`
in as the seller's latest answer (and `asked`, if given, as the question before it). Cases with the same "pair" are
a minimal pair (the right sign differs between them); the summary then also counts pairs where both are right,
pairs where the model gives both the same sign, and pairs ordered right (the better case gets the higher p(+) - p(-)). Other fields of a case (such as "suggest") are not read. This is a diagnostic: nothing it finds goes into the game (the project rule: only pre-training and SFT make a model better), and a
result that points at the word's representation is fixed in the data, the tokenizer or the pre-training.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch

ML = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ML)]
SIGNS = ("+", "0", "-")
TEMPERATURE = 0.7                   # as the game samples (eval_buyer_b1.TEMPERATURE)
RANDOM_DRAWS = 20
UNRELATED = "table"
SET_DIR = ML / "runs/eval_sets_b3n"
SELLER = "Seller: "
CONDITIONS = ("base", "text_swap", "donor", "pieces_mean", "context_mean", "unrelated", "random", "negated")


def with_answer(prompt: str, answer: str, asked: str | None = None) -> str:
    """The prompt with the seller's latest answer replaced (the line ends `<|end|><|react|>`), and with `asked` the
    question on the line before it, so the answer still answers what was asked."""
    lines = prompt.split("\n")
    last = max(i for i, line in enumerate(lines) if line.startswith(SELLER))
    lines[last] = SELLER + answer + lines[last][lines[last].index("<|end|>"):]
    if asked is not None:
        who, sep, _ = lines[last - 1].partition(" asked: ")
        assert sep, "the line before the seller's answer is not a question"
        lines[last - 1] = who + sep + asked
    return "\n".join(lines)


def last_answer(prompt: str) -> str:
    """The seller's latest answer in the prompt."""
    line = [line for line in prompt.split("\n") if line.startswith(SELLER)][-1]
    return line[len(SELLER):line.index("<|end|>")]


def find_pieces(ids: list[int], word_ids: list[int]) -> list[int]:
    """Positions of the last place where the word's pieces stand in the prompt's ids."""
    n = len(word_ids)
    for start in range(len(ids) - n, -1, -1):
        if ids[start:start + n] == word_ids:
            return list(range(start, start + n))
    raise ValueError("the word's pieces are not in the prompt as one run; check the pieces printed for the case")


@contextmanager
def swapped_inputs(model, vectors: dict[int, torch.Tensor]) -> Iterator[None]:
    """While active, the model reads `vectors[position]` instead of the embedding of the token at that position."""
    def hook(_module, _inputs, out):
        out = out.clone()
        for position, vector in vectors.items():
            out[:, position] = vector
        return out
    handle = model.embed.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def sign_probs(model, ids: list[int], sign_ids: list[int], vectors: dict[int, torch.Tensor] | None = None) -> dict:
    """Probability of each sign as the first token after the prompt (softmax at the game's temperature over the whole
    vocabulary, then the three signs scaled to add to 1), and the mass the three signs had before that."""
    with swapped_inputs(model, vectors or {}):
        logits, _ = model(torch.tensor([ids]))
    p = torch.softmax(logits[0, -1].float() / TEMPERATURE, -1)[sign_ids]
    return {"mass": float(p.sum()), **{s: float(x) for s, x in zip(SIGNS, p / p.sum())}}


def sign_ids(tok) -> list[int]:
    """The id each sign gets as the first token of a target ("+ | …")."""
    got = [tok.encode(f"{s} | x")[0] for s in SIGNS]
    assert len(set(got)) == len(SIGNS), "the signs must start with different tokens"
    return got


def spoken(tok, word: str) -> list[int]:
    """The ids of a word in the middle of a sentence: the tokenizer adds no space in front of a text's first word,
    so the pieces of "long" alone are not the pieces of " long" after another word."""
    return tok.encode(" " + word)


def pieces(tok, ids: list[int]) -> list[str]:
    """The pieces behind ids, for reading."""
    return [tok.sp.id_to_piece(i) if i < tok.sp_size else tok.extra_text[i] for i in ids]


@torch.no_grad()
def run_case(model, tok, case: dict) -> list[dict]:
    """One row per condition: the sign probabilities, the sign it picks, p(want), and the size of the vectors it put in."""
    prompt, word, donor = case["prompt"], case["word"], case["donor"]
    ids, word_ids = tok.encode(prompt), spoken(tok, word)
    places = find_pieces(ids, word_ids)
    table = model.embed.weight.detach()
    typical = float(table.norm(dim=1).mean())
    signs = sign_ids(tok)
    own = {p: table[ids[p]] for p in places}
    answer_ids = [i for i in tok.encode(last_answer(prompt)) if i not in word_ids]
    plans: dict[str, tuple[list[int], dict[int, torch.Tensor]]] = {
        "base": (ids, {}),
        "text_swap": (tok.encode(donor.join(prompt.rsplit(word, 1))), {}),
        "donor": (ids, {p: table[spoken(tok, donor)].mean(0) for p in places}),
        "pieces_mean": (ids, {p: table[word_ids].mean(0) for p in places}),
        "context_mean": (ids, {p: table[answer_ids].mean(0) for p in places}),
        "unrelated": (ids, {p: table[spoken(tok, case.get("unrelated", UNRELATED))].mean(0) for p in places}),
        "negated": (ids, {p: -v for p, v in own.items()}),
    }
    rows = []
    for name in CONDITIONS:
        if name == "random":
            draws = []
            for seed in range(RANDOM_DRAWS):
                g = torch.Generator().manual_seed(seed)
                v = torch.randn(table.size(1), generator=g)
                draws.append(sign_probs(model, ids, signs, {p: v * typical / v.norm() for p in places}))
            probs = {k: sum(d[k] for d in draws) / RANDOM_DRAWS for k in draws[0]}
            norm = typical
        else:
            row_ids, vectors = plans[name]
            probs = sign_probs(model, row_ids, signs, vectors)
            norm = float(sum(v.norm() for v in vectors.values()) / len(vectors)) if vectors else float("nan")
        pick = max(SIGNS, key=lambda s: probs[s])
        rows.append({"case": case["name"], "pair": case.get("pair"), "condition": name, "want": case["want"],
                     "pick": pick,
                     "p_want": probs[case["want"]], "mass": probs["mass"], **{f"p{s}": probs[s] for s in SIGNS},
                     "norm_ratio": norm / typical})
    return rows


def summarise(rows: list[dict]) -> str:
    """Per condition over all cases: mean p(want), cases where the pick is `want`, and the mean change of p(want) from base."""
    base = {r["case"]: r["p_want"] for r in rows if r["condition"] == "base"}
    out = [f"{'condition':<13}{'mean p(want)':>13}{'pick = want':>13}{'change vs base':>16}{'norm ratio':>12}"]
    for name in CONDITIONS:
        got = [r for r in rows if r["condition"] == name]
        if not got:
            continue
        n = len(got)
        norm = "-" if name in ("base", "text_swap") else f"{sum(r['norm_ratio'] for r in got) / n:.2f}"
        out.append(f"{name:<13}{sum(r['p_want'] for r in got) / n:>13.3f}{sum(r['pick'] == r['want'] for r in got):>10}/{n}"
                   f"{sum(r['p_want'] - base[r['case']] for r in got) / n:>+16.3f}{norm:>12}")
    return "\n".join(out)


def pair_summary(rows: list[dict]) -> str:
    """Per condition, over the pairs whose two cases are both in the rows: pairs with both picks right, pairs with the
    same pick in both (a model that ignores the turning half), and pairs ordered right: the case whose `want` is higher
    (+ over 0 over -) gets the higher p(+) - p(-). The last needs no pick, so a buyer who holds most of the time (63%
    of the reactions in the new validation set) can still show it reads the turn. Chance: both right 1/9, ordered 1/2."""
    pairs = sorted({r["pair"] for r in rows if r["pair"]})
    if not pairs:
        return ""
    out = [f"{'condition':<13}{'both right':>12}{'same pick':>12}{'ordered':>10}   ({len(pairs)} pairs)"]
    for name in CONDITIONS:
        got = {p: [r for r in rows if r["condition"] == name and r["pair"] == p] for p in pairs}
        got = {p: v for p, v in got.items() if len(v) == 2}
        right = sum(all(r["pick"] == r["want"] for r in v) for v in got.values())
        same = sum(v[0]["pick"] == v[1]["pick"] for v in got.values())
        rank = {s: i for i, s in enumerate(reversed(SIGNS))}                 # "-" 0, "0" 1, "+" 2
        differ = [v for v in got.values() if rank[v[0]["want"]] != rank[v[1]["want"]]]
        ordered = sum((v[0]["p+"] - v[0]["p-"] > v[1]["p+"] - v[1]["p-"]) == (rank[v[0]["want"]] > rank[v[1]["want"]])
                      for v in differ)
        out.append(f"{name:<13}{right:>9}/{len(got)}{same:>9}/{len(got)}{ordered:>7}/{len(differ)}")
    return "\n".join(out)


def load_cases(path: Path, sets: Path) -> list[dict]:
    """The cases of the file, with the prompt built for the ones that name an eval example."""
    cases = []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        c = json.loads(line)
        if "from" in c:
            which, index = c["from"]
            [file] = sets.glob(f"val_{which}_format*.jsonl")
            ex = json.loads(file.read_text().splitlines()[index])
            assert ex["task"] == "react", f"{file.name} line {index} is a {ex['task']} example"
            c["prompt"] = with_answer(ex["prompt"], c["answer"], c.get("asked"))
        c.setdefault("name", f"case{i}")
        cases.append(c)
    blank = [c["name"] for c in cases if c["want"] not in SIGNS]
    if blank:
        raise SystemExit(f"`want` must be one of {SIGNS}; fill it in for: {', '.join(blank)}")
    return cases


def main() -> None:
    """Run every case on the checkpoint and write one row per case and condition."""
    from sellanything_ml.llm.evaluate import load_model
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--sets", type=Path, default=SET_DIR, help="folder of val_{old,new}_format*.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    model, tok = load_model(a.checkpoint, torch.device("cpu"))
    rows = []
    for case in load_cases(a.cases, a.sets):
        print(f"{case['name']}: {case['word']!r} = {pieces(tok, spoken(tok, case['word']))}, donor "
              f"{case['donor']!r} = {pieces(tok, spoken(tok, case['donor']))}")
        rows += run_case(model, tok, case)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(summarise(rows))
    if pair_summary(rows):
        print("\n" + pair_summary(rows))


if __name__ == "__main__":
    main()
