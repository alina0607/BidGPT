# Related work

Literature review for Sell Anything v2 and the thesis (2026-10-05). Every paper below was read in full
unless the "Read" column says otherwise; numbers are quoted from the papers with their table or
section. What we take from each paper is applied in [`design-v2.md`](design-v2.md) and marked there
with the paper's name.

Search limits for the novelty check: WebSearch, the arXiv API and publisher pages. Semantic Scholar
returned HTTP 429 and ACM DL returned 403 throughout, and Google Scholar was not scraped, so
"not found" below means "not found in those sources".

## 1. Summary

| Paper | Area | Overlap with our thesis claim | What we adopt |
|---|---|---|---|
| Ha & Eck 2017, Sketch-RNN | stroke drawing generation | none | end-of-drawing handling, temperature ranges, choose by recognisability not loss |
| Ribeiro et al. CVPR 2020, Sketchformer | stroke Transformer, token dictionary | none | K-means pen-move dictionary (paper procedure, not released code) |
| Seddati et al. CBMI 2016, DeepSketch 2 | partial-sketch recognition | none | fine-tune from the full-sketch model; per-completeness evaluation |
| Huang & Canny UIST 2019, Sketchforme; Huang et al. IUI 2020, Scones | sketched scenes from text | partial: generated sketches can help users (language learning) | object sizes, no stretching, no-overlap test, study design |
| He et al. EMNLP 2018 | negotiation dialogue, strategy vs generation | partial: decoupling decisions from text | rule baselines, degeneracy metric, speech-vs-decision consistency, event log |
| Kim et al. IJCAI 2025, MART | LLM merchant NPC | partial: LLM trader NPC | money-impact metric, seller audit, paired tests |
| Lei 2026, PrefBench (preprint) | hidden-preference pricing benchmark | none | downstream metric for inferred preferences, baselines, bootstrap CIs |
| Anderson et al. IJCAI 2019 | explaining agents to lay people in a game | partial: explanation → understanding, in a game | study design, difficulty stratification |
| Amitai et al. AAAI 2024, COViz | outcome views + value bars as explanations | partial–strong on mechanism | tag-bar comparison condition |
| Ni et al. 2026, PEARL (preprint) | personalised lifestyle scenes of a product | **strong on the scene idea**, none on its purpose | do not claim the scene concept as new |
| Jiang et al. IJCNLP-AACL 2025, HARBOR | persona bidders in auctions | partial: persona-driven bidders | — |
| Kim et al. 2025, SIBP | LLM trading NPC, code computes prices | partial: architecture | citation for D2 |
| Ehsan et al. IUI 2019 | generated rationales for an agent in Frogger | partial | perception items for the buyer speech |
| Hase & Bansal ACL 2020 | do explanations help users predict a model? | none (method) | **the core study protocol** |
| Yan et al. SIGIR 2023, Personalized Showcases | visual explanations for recommendations | partial | — |

## 2. Sketch generation and recognition

**Sketch-RNN** (Ha & Eck, arXiv:1704.03477). Sequence-to-sequence VAE over stroke-5 points with a
Gaussian-mixture output; trained per class (70K drawings each). Supp §4: one unconditioned model over
75 classes mixes classes, and the authors propose "class information outside the latent space" as
the fix (which is our class token). Supp §6: lower reconstruction loss does not guarantee better
drawings. Stopping is called "a key challenge" (§3.2). No recognisability metric, user study or
copy check.

**Sketchformer** (Ribeiro et al., CVPR 2020, arXiv:2002.10381). Transformer autoencoder; best input is
a K-means dictionary of K = 1,000 pen moves from 100k sampled moves, 20% of them pen-up (§3.1).
Table 1 (345 classes, 862k test sketches): Tok-Dict 78.34, continuous 77.68, LSTM baselines 72.93 /
67.69; shuffling stroke order costs only 0.73–1.68 points. In reconstruction user studies the
continuous Transformer was never preferred (0.00, Table 2) and Tok-Dict won on long sketches (51.4).
No numeric reconstruction error is reported for the dictionary (Fig. 2 is qualitative). The released
code differs from the paper (5M samples; the 20% pen-up resampling is broken or skipped), so we
follow the paper.

**DeepSketch 2** (Seddati, Dupont, Mahmoudi, CBMI 2016; read from the author's upload). Partial
sketches = first 20/40/60/80/100% of strokes. A single network fine-tuned on all levels ("merged")
lifted 20% → 28.33% and 40% → 50.72% top-1, against 4.66% and 21.55% for the full-sketch model, and
kept full-sketch accuracy (75.01 vs 75.02; Tables II, VI). A completeness auxiliary head adds +3.79
at 20% (Table VII). Fine-tuning the whole network beats fine-tuning the head at low completeness.

## 3. Sketched scenes

**Sketchforme** (Huang & Canny, UIST 2019, arXiv:1904.04399) and **Scones** (Huang et al., IUI 2020,
arXiv:2005.07781). Layouts are learned from Visual Genome (101,968 relations) or CoDraw (9,993
sessions); objects are drawn by per-class Sketch-RNNs conditioned on aspect ratio (and a mask in
Scones). Overlap and occlusion are not handled (stated limitation); forcing a square ratio drew only
an elephant's face (Fig. 14b). A language-learning study (MTurk, 38 participants, between subjects)
found learners with sketches faster (246 s vs 338 s, p = 0.011) but **not more accurate**
(p = 0.132). Scones users who asked for classes it could not draw gave most of the low enjoyment
scores. For us a learned layout would learn physical co-occurrence from photos, which is the wrong
thing for symbolic props (a crown for status), so we keep rule-based zones.

## 4. Negotiation agents and trader NPCs

**He et al.** (EMNLP 2018, arXiv:1808.09637). CraigslistBargain (6,682 dialogues). A rule parser maps
text to coarse acts, a manager picks the next act, a retrieval generator says it. Best bot: learned
act + hand rules for prices (human-likeness 3.6 / 5, Table 7). RL on prices learned to insist and
frustrated people (agreement 0.22); RL on words collapsed (top-3 sentences = 81.6%, 100%, 100% of
output). The parser's accuracy is never reported. Partner was a simulated model, which the authors
say misled RL.

**MART** (Kim, Kim, Seo, Kim, IJCAI 2025, arXiv:2412.11189). LLM appraiser + negotiator for a WoW-item
merchant; KD from Llama 3.1 405B into 1B/8B. Every small model made arithmetic errors and the 1B
model invented items (§5.6). No human players: the player is GPT-4o. Only one objective price per
item, one merchant, no imagery.

**SIBP** (Kim et al., KDD'25 workshop, arXiv:2507.07203). A merchant NPC writes `__PRICE__` and game
code fills it in: price accuracy 100% / 99.7% with placeholders vs 79.7% / 82.3% without (Table 2).

**HARBOR** (Jiang, Xiong, Liu, IJCNLP-AACL 2025, arXiv:2502.12149). LLM bidders with buyer personas in
house auctions; agents infer rivals' persona vectors in [−1, 1] from bids. No humans, no imagery.

**PrefBench** (Lei, arXiv:2605.22855, preprint). An LLM seller prices a car bundle for a simulated buyer
whose hidden preferences are a non-negative weight vector over 5 channels that sums to 1. The seller
cannot ask questions; only profit is scored. Models closed in round 1 in 7,408–7,500 of 7,500
episodes, and profit was no better than random within overlapping CIs (Table 3, App. D).

## 5. Explaining agents to people

**Anderson et al.** (IJCAI 2019, arXiv:1903.09708). RTS game; the agent's value is split into 6 named
reward types shown as bars. N = 124, between subjects. Participants predicted each action before
seeing the explanation; reward bars improved mental-model scores (F = 6.454, p = .0123). On
counter-intuitive decisions accuracy was ≤ 25% in every condition; on easy ones control already
reached ≥ 75%. "Everything" caused overload (17 timeouts vs 10).

**COViz** (Amitai, Septon, Amir, AAAI 2024, arXiv:2312.11118). Outcome videos of the chosen vs
second-best action, plus reward bars, to show which reward an agent cares about. Bars + outcomes
.64 correct vs bars .45, outcomes .42, summary .31 (Study 1, N = 115). **Outcomes alone were not
better than bars alone.** People ranked the combination last (53%).

**Hase & Bansal** (ACL 2020). Forward and counterfactual simulation tests: predict a model's output
before and after explanations, with explained and test items kept apart. Only one of five methods
helped (LIME, tabular, +11.25 points). Subjective ratings did not predict correctness (§5.3).

**Ehsan et al.** (IUI 2019, arXiv:1901.03729). A learned model generates natural-language rationales
for a Frogger agent; rated on confidence, human-likeness, justification, understandability. Measured
perception, not prediction.

**PEARL** (Ni et al., arXiv:2610.00737, 30 Sep 2026, preprint; cite by arXiv id, the same first
author has another 2026 paper). "Personalized Scene Generation": place a given product in a scene
that reflects a user's lifestyle (a flashlight on a hiking trail vs in an engine bay). A VLM planner
reads the user's Amazon reviews, picks **one identity element with a one-sentence `<why>`**
(App. F), writes a background prompt, SDXL inpaints around the pasted product, and a reflector
revises it. Evaluated by a CLIP retrieval probe (H@5 .2297 vs .2162), an aesthetic score, a
two-model MLLM judge, and **one annotator** on 25 scene and 25 creative pairs (19/25 and 16/25).
Its ablation replaces DPO with SFT but never removes the reflector. Purpose: engagement. Not an
explanation, not a game, no study of understanding.

**Personalized Showcases** (Yan et al., SIGIR 2023). Retrieved images + generated text explain a
recommendation; rated for expressiveness, not understanding.

Also read, used for methods: Hoffman et al. 2018 (Explanation Satisfaction Scale, App. C);
Mohaddesi et al. CHI 2023 (Thought Bubbles: eliciting the player's mental model in game);
Stanley, Zhang, Lewis 2025 preprint (Bayesian models of how people infer bots' preferences).

## 6. What is and is not new

Already done; we cite these and do not claim them:

1. An NPC trader whose dialogue is separate from deterministic price code (MART, SIBP, He et al.).
2. Persona-driven bidders with hidden preferences (HARBOR, PrefBench).
3. Testing whether a decomposition of an agent's values helps people understand or predict it, in a
   game (Anderson et al., COViz).
4. A generated scene that places the same object in each person's lifestyle, planned from an
   inferred preference with a one-sentence reason for the chosen element (PEARL, App. F).
6. "The language model only speaks; other code decides": CICERO (Meta FAIR, Science 2022, as
   cited by Gallotta et al.) and Gallotta et al. §4.
7. Small models learning a narrow skill from LLM-written data (TinyStories; phi-1); distilling a
   large model into smaller ones by SFT on its outputs (DeepSeek-R1).
5. Generated sketches that help people with a task (Sketchforme); stroke sketch generation
   (Sketch-RNN, Sketchformer).

Not found in any source searched:

- (a) A generated scene used as a **player-facing explanation of an NPC's hidden valuation**, tested
  with players for understanding or prediction.
- (b) **Stroke-drawn doodles** as the explanation medium (prior work uses bars, saliency maps, videos,
  photos, diffusion images).
- (c) **One small signed tag space** that drives the valuation, the dialogue and the scene, so the
  three agree by construction; each prop's reason is a term of the same score that sets the bid
  (checkable), not a free-text justification as in PEARL.
- (d) For thesis idea B: generalisation of a 13–28M model to **held-out item categories**, split by
  similarity to training (near / far), with exact engine labels. TinyStories never holds out topics,
  phi-1's gains concentrate on near-duplicate problems (its Table 3), and DeepSeek-R1's smallest
  student is 1.5B and is tested only on new problems from the same domains.

Draft claim:

> Each NPC buyer's hidden preferences are one signed 8-tag vector that drives, by construction, a
> deterministic valuation, the buyer's dialogue (a 12.9M-parameter Transformer language model), and a
> per-buyer sketch scene drawn stroke by stroke that shows the item in that buyer's imagined home;
> every prop's stated reason is a term of the same score that sets the bid. Prior work generates
> personalised lifestyle scenes of a product, with a free-text reason for the chosen scene element,
> to improve engagement (Ni et al., arXiv:2610.00737), infers negotiators' preferences from dialogue
> (Chan et al., 2024), and explains agent preferences with value bars or outcome videos (Anderson et
> al., 2019; Amitai et al., 2024). To our knowledge, generated scenes have not been used as
> player-facing explanations of an NPC's valuation or tested for whether they help people predict
> its behaviour. In a controlled player study we test whether these scenes improve players' accuracy
> in predicting which buyer values an item more, and how a changed attribute moves a bid, compared
> with dialogue alone, tag bars, and scenes with random props.

Two cautions. (1) COViz and Hase & Bansal both found that one extra explanation often does not beat
a reasonable baseline, and Sketchforme found speed but not accuracy gains, so a null result for
scenes alone is plausible; the primary metric must be fixed in advance. (2) The claim needs the bids
to really come from the tags: if bids come from the buyer GPT's Δ (held-out sign accuracy 76.18%,
`ml/runs/sft_v5/final_v5.json`), scenes could "explain" decisions the buyer did not make. See
design-v2 §3.2.

## 7. Second reading batch (2026-10-05)

What each paper changed is listed in `design-v2.md` §13; full notes come from the reading agents.

| Paper | One-line takeaway for us |
|---|---|
| Eldan & Li 2023, TinyStories | 1–35M models write coherent, consistent text when the domain is narrow and the LLM-written data is diversified with random constraints; topics are never held out |
| Gunasekar et al. 2023, phi-1 | filtered + synthetic "textbook" data and a small synthetic fine-tuning set make a 1.3B model strong; its HumanEval lead is mostly on problems with near matches in the fine-tuning data (Table 3) |
| Gururangan et al. 2020, Don't Stop Pretraining | continued pre-training on domain text helps that domain; text from another domain usually hurts, and task-to-task transfer within a domain loses 0.2–7.7 F1 |
| Xie et al. 2023, DSIR | importance resampling on hashed n-grams selects domain data better than classifiers or top-k (avg F1 83.76 vs 82.88–83.65); KL reduction predicts results (r = 0.82); CPU-cheap |
| Penedo et al. 2024, FineWeb | FineWeb-Edu = Llama-3-70B labels (460k) → linear head on embeddings, threshold 3; strong on MMLU/ARC at 1.71B; down-samples commerce topics |
| Muennighoff et al. 2023 | up to ~4 epochs of repetition costs little; effective data decays with a half-life of ~15 epochs (R*_D = 15.39) |
| DeepSeek-AI 2025, R1 (distillation) | SFT on ~800k teacher samples distils into 1.5–70B students; distillation beats RL at the same size; no held-out categories |
| Hinton et al. 2015 | distillation = matching soft targets at temperature T; needs teacher logits, which we do not have — ours is data-level distillation with engine labels |
| Du et al. 2023/ICML 2024, Multiagent Debate | 3 copies of one model, 1 debate exchange, beat single answers; vs majority vote the GSM8K gap (85.0 vs 81.0, n = 100) is within noise; copies share confident errors |
| Gebru et al. 2021, Datasheets | 57 questions in 7 lifecycle sections; written by people, one per dataset version |
| Park et al. 2023, Generative Agents | memory stream + reflection + planning; believability ranked by 100 people; 1.3% hallucinated relationship claims; agents overly cooperative |
| Gallotta et al. 2024, LLMs and Games survey | 9 roles of LLMs in games; venues CoG, FDG, AIIDE, ToG; "LLM" means ≥ a few hundred million parameters; on-device cost and grounding are open problems |
| Chan et al. 2024, NegotiationToM | desire/belief/intention tracking in negotiations; GPT-4 63% vs humans 91% on desires; main error is naming an item when the answer is "not given" |
| Ni et al. 2026, PEARL (full read) | see §5; reason per chosen element exists, so our novelty is a reason that is a term of the bid score |

## 8. Third reading batch (2026-10-06): the papers closest to the next work

Read by agents before wiring the drawing GPT into the game and starting the buyer GPT; full notes are
kept with the session. "Read" says how much of the original text the agent saw: WebFetch returns a
model-written summary, so papers read only that way are marked ◐ in `reading-list.md`, and numbers
below come from tables or pages the agent checked.

| Paper | Read | Takeaway for us |
|---|---|---|
| **Painter** (Pourreza et al., arXiv 2308.08520, 2023 preprint) | HTML via summaries, 7 targeted passes | OPT-125M/1.3B writes strokes as plain-text coordinates and sees the canvas through a ResNet-50. Generation is **never measured** ("an extensive user-study is out of budget", figures are selected); classification 3.20–40.26% across runs (Table 3), no baselines. Nothing on stopping beyond an end tag, temperature, copying or speed. Our classifier-judged evaluation of 3,450 drawings, with non-ending and copy rates, goes beyond it. |
| **Kwon et al.** (Findings of EMNLP 2024, pp. 5391–5413) | full PDF | 35 tasks on four multi-issue datasets (**not** CraigslistBargain: "CRA" is DeVault et al. 2015, 119 dialogues). Fine-tuned Flan-T5-base parses proposals perfectly on DND (1.0 vs GPT-4 0.866) but everyone is weak on the 119-dialogue CRA (0.439 / 0.369, Table 9). Arithmetic fails without CoT (deal total 0.083 → 0.714, Fig. 9c); subjective judgements correlate weakly (PCC 0.3–0.39). Partner-preference inference improves with turns (0.42 → 0.75) while own-preference accuracy falls (0.94 → 0.75, Fig. 6). Failure modes: re-proposing rejected offers, over-agreeable acceptance. One multi-task Flan-T5 did worse than per-task models (App. B, no numbers). |
| **Xia et al.** (Findings of ACL 2024, pp. 3579–3602) | HTML via summaries, tables row by row | 930 Amazon products, asymmetric budget/cost game. Every LLM *buyer* loses money (GPT-4 SNP −33.81) while sellers profit (Table 2); buyers break their budget and open at 0.81–1.00 of it (Table 3); bigger Llama-2 is no better as a buyer. **OG-Narrator**: a fixed offer schedule p = (0.5 + 0.5·t/tₘ)·B plus an LLM that only narrates turns Phi-2 from −1.58 to 2015.98 SNP (Table 4). This is our engine/voice split. Narration fidelity (do the words match the action) is never measured. |
| **τ-bench** (Yao et al., arXiv 2406.12045, 2024 preprint) | full PDF | A simulated user (gpt-4, temperature 1) with a hidden instruction; success = final database equals ground truth and required strings appear; **pass^k** = all k of n trials succeed, E[C(c,k)/C(n,k)]. gpt-4o retail falls from >60% (pass^1) to <25% (pass^8) (Fig. 4). Of 40 failures, 4 were the simulated user's instructions (§5.2); simulator reliability is never measured. Text is never checked for contradictions. |
| **G-Eval** (Liu et al., EMNLP 2023, pp. 2511–2522) | full PDF | GPT-4 judge with auto-written steps and probability-weighted scores: SummEval ρ 0.514 vs a bare prompt's 0.479 (Table 1); best agreement with people is ~0.5–0.6. The judge preferred GPT-written summaries even where people preferred the human ones (§4, Fig. 2; annotators' α = 0.07). |
| **Distillation Scaling Laws** (Busbridge et al., ICML 2025) | full HTML, App. A–G | Logit distillation (τ = 1) beats supervised training only below a token/compute threshold that grows with student size (§5.1), and never if the teacher is trained for one student (§5.3); student loss is U-shaped in teacher strength (capacity gap, Fig. 4). Synthetic-text ("hard") distillation and a teacher trained on other data are **explicitly out of scope** (App. A), so none of the fitted numbers transfer to our LLM-written data. Smallest model 143M. |
| **nanochat** (Karpathy, repo at `92d63d4`, 2026-07-03, and Discussion #1) | code and discussion via summaries | Mid-training was **removed** on 2026-01-31 once every row starts at BOS and is never cropped; its mixture moved into SFT. Chat/tool special tokens are **reserved in the tokenizer before pre-training** (only BOS is seen then). SFT trains on assistant tokens only, masks tool output, warm-starts the optimizer at 0.8× LR. Tokens per parameter ≈ 10 (matrices + lm_head) for the speedrun; the MacBook script trains d6 on ~82M tokens in ~30 min and says "You will not get far on your Macbook". |

