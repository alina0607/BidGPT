# Reading list

Every paper mentioned while designing v2 and the thesis (2026-10-05); E14–E20 and sections F–H were added
2026-10-10 from the cat landmarks, the cat painter and the buyer dialogue data. The owner's rule: **all of them
are read in full before implementation starts**, and every concept we take is written into
[`design-v2.md`](design-v2.md) with the paper's name. Summaries of papers already read are in
[`related-work.md`](related-work.md).

Status: ✅ read in full · ◐ read in part · ☐ not read yet. "Tier" is the reading order: 1 shapes the
design or the thesis directly, 2 supports a specific decision, 3 is background.
Citation details of ☐ papers are as remembered or as cited by another paper, and are verified when
the paper is read.

## A. Sketch recognition and generation

| # | Paper | Venue | Why it matters to us | Tier | Status |
|---|---|---|---|---|---|
| A1 | Ha & Eck, A Neural Representation of Sketch Drawings (Sketch-RNN) | arXiv 1704.03477, 2017 | stroke generation, stopping, temperature | 1 | ✅ |
| A2 | Ribeiro et al., Sketchformer | CVPR 2020 | pen-move dictionary, stroke Transformer | 1 | ✅ |
| A3 | Seddati et al., DeepSketch 2 (partial sketches) | CBMI 2016 | live guessing (F1) | 1 | ✅ |
| A4 | Huang & Canny, Sketchforme | UIST 2019 | sketched scenes, user study | 1 | ✅ |
| A5 | Huang et al., Scones | IUI 2020 | conversational scenes, user study | 1 | ✅ |
| A6 | Pleskov & Ostyakov, Quick, Draw! 1st place write-up | Kaggle 2018 | CNN vs RNN, stacking | 2 | ✅ |
| A7 | Google, Recurrent Neural Networks for Drawing Classification (TF tutorial) | 2017 | our classifier baseline | 2 | ✅ (devlog 2026-10-04) |
| A8 | Painter: Teaching Auto-regressive Language Models to Draw Sketches | arXiv 2308.08520 | an LLM drawing strokes | 2 | ◐ |
| A9 | Hu, Li, Qi & Song, Scale-Adaptive Diffusion Model for Complex Sketch Synthesis (code `HuJijin/Adaptive_Guided_Sketch`) | ICLR 2024 | pixel diffusion beats vector models on QuickDraw FID (Table 1: Sketch-RNN 8.15, theirs 3.08); classifier-free guidance (CFDG) as a strong baseline; too much guidance "over-sketches". Sent by the owner 2026-10-07; used for drawing GPT v2's guidance and its stroke-count check (devlog 2026-10-07) | 2 | ✅ (main paper, 20 pp, read 2026-10-07) |
| A10 | Li, Hu, Chen, Yang, Qi & Song, SACG++: Complex Sketch Generation via Representation-Enhanced Scale-Adaptive Classifier Guidance | IJCV 2026, doi 10.1007/s11263-026-02768-5 | journal extension of A9 (title and authors from Crossref); the publisher page needs a login, so not read | 3 | ☐ |

## B. Negotiation, trading NPCs, preferences

| # | Paper | Venue | Why | Tier | Status |
|---|---|---|---|---|---|
| B1 | He et al., Decoupling Strategy and Generation in Negotiation Dialogues | EMNLP 2018 | CraigslistBargain; decisions vs text | 1 | ✅ |
| B2 | Kim et al., MART: LLMs for Active Merchant NPCs | IJCAI 2025 | trader NPC, small models, KD | 1 | ✅ |
| B3 | Lei, PrefBench | arXiv 2605.22855, 2026 | hidden-preference pricing | 1 | ✅ |
| B4 | Kim et al., State-Inference-Based Prompting (SIBP) | KDD'25 workshop | code fills prices | 2 | ✅ |
| B5 | Jiang et al., HARBOR | IJCNLP-AACL 2025 | persona bidders | 2 | ✅ |
| B6 | Chan et al., NegotiationToM | Findings of EMNLP 2024, pp. 4211–4241 | inferring desires in negotiation | 1 | ✅ |
| B7 | Kwon et al., Are LLMs Effective Negotiators? | Findings of EMNLP 2024 | negotiation evaluation | 2 | ✅ |
| B8 | Xia et al., Measuring Bargaining Abilities of LLMs | Findings of ACL 2024 | buyer-side bargaining | 2 | ◐ |
| B9 | Hua et al., Assistive LLM Agents for Socially-Aware Negotiation | Findings of EMNLP 2024 | negotiation | 3 | ☐ |
| B10 | Shea et al., ACE: an LLM-based Negotiation Coaching System | EMNLP 2024 | negotiation | 3 | ☐ |
| B11 | Jin et al., Persuading across Diverse Domains | ACL 2024 | persuasion data | 3 | ☐ |
| B12 | Yao et al., τ-bench | arXiv 2406.12045, 2024 | simulated user with hidden info | 2 | ✅ |
| B13 | Wu et al. / AucArena (LLM auctions) | arXiv 2310.05746, 2023 | auction agents | 3 | ☐ |
| B14 | Park & Lee, Exploring the value of purchasing online game items | Computers in Human Behavior 2011 | item value dimensions vs our tags | 3 | ☐ |
| B15 | Guo & Barnes, Virtual item purchase behavior | Electronic Commerce Research 2009 | item value | 3 | ☐ |

## C. Explaining agents to people; study design

| # | Paper | Venue | Why | Tier | Status |
|---|---|---|---|---|---|
| C1 | Anderson et al., Explaining RL to Mere Mortals | IJCAI 2019 | explanation study in a game | 1 | ✅ |
| C2 | Amitai et al., COViz | AAAI 2024 | outcome views + value bars | 1 | ✅ |
| C3 | Hase & Bansal, Evaluating Explainable AI | ACL 2020 | the core study protocol | 1 | ✅ |
| C4 | Ni et al., PEARL (personalized scenes) | arXiv 2610.00737, 2026 | closest to buyer scenes | 1 | ✅ |
| C5 | Ehsan et al., Automated Rationale Generation | IUI 2019 | generated rationales in a game | 2 | ✅ |
| C6 | Yan et al., Personalized Showcases | SIGIR 2023 | visual explanations | 3 | ✅ |
| C7 | Hoffman et al., Metrics for Explainable AI | arXiv 1812.04608 | satisfaction scale | 2 | ◐ |
| C8 | Mohaddesi et al., Thought Bubbles | CHI 2023 | eliciting players' mental models | 2 | ✅ |
| C9 | Stanley, Zhang, Lewis, Inferring Hidden Motives | arXiv 2511.07825 | how people infer bots' preferences | 3 | ◐ |
| C10 | Juozapaitis et al., reward decomposition | 2019 | value bars | 3 | ☐ |
| C11 | Septon et al. | arXiv 2210.11825 | explanation study | 3 | ☐ |
| C12 | Newn et al., gaze-based intent (CHI PLAY 2017) | 2017 | intent visualisation | 3 | ☐ |
| C13 | Villareale, Fox, Zhu, Can Games Be AI Explanations? | DiGRA 2024 | games as explanations | 2 | ☐ |
| C14 | Czapp et al. (RecSys 2024) and AdBooster | RecSys 2024 pp. 768–770 (verified via PEARL) / 2023 | product background generation | 3 | ☐ |

## D. LLMs in games (is a game-based thesis "real research"?)

| # | Paper | Venue | Why | Tier | Status |
|---|---|---|---|---|---|
| D1 | Park et al., Generative Agents: Interactive Simulacra of Human Behavior | UIST 2023 | LLM characters in a game world; evaluation of believability | 1 | ✅ |
| D2 | Wang et al., Voyager | arXiv 2305.16291, 2023 | LLM agent in Minecraft | 2 | ☐ |
| D3 | Baker et al., Video PreTraining (VPT) | NeurIPS 2022 | Transformer trained for Minecraft | 2 | ☐ |
| D4 | Gallotta et al., Large Language Models and Games: A Survey and Roadmap | IEEE Transactions on Games, early access 2024 | map of the field | 1 | ✅ |
| D5 | Christiansen et al., Presence with LLM-driven NPCs | VRST 2024 | player study with LLM NPCs | 2 | ☐ |
| D6 | Marincioni et al., LLM NPC emotional states and player emotions | IEEE CoG 2024 | player study | 2 | ☐ |
| D7 | Cox & Ooi, Conversational interactions with NPCs | CONVERSATIONS 2023 (2024) | player feedback guidelines | 3 | ☐ |
| D8 | Peng et al., Player-driven emergence in LLM-driven game narrative | IEEE CoG 2024 | | 3 | ☐ |
| D9 | Phillips, Lang, Mould, Goal-oriented interactions in games using LLMs | IEEE Transactions on Games 2024 | | 3 | ☐ |
| D10 | Buongiorno et al., PANGeA | AIIDE 2024 | | 3 | ☐ |

## E. Small models, data selection, distillation (thesis idea B)

| # | Paper | Venue | Why | Tier | Status |
|---|---|---|---|---|---|
| E1 | Eldan & Li, TinyStories | arXiv 2305.07759, 2023 (v2) | how small a model can be in a narrow domain | 1 | ✅ |
| E2 | Gunasekar et al., Textbooks Are All You Need (phi-1) | arXiv 2306.11644, 2023 (v2) | focused high-quality data for small models | 1 | ✅ |
| E3 | Gururangan et al., Don't Stop Pretraining | ACL 2020 | domain-adaptive pre-training | 1 | ✅ |
| E4 | Xie et al., Data Selection via Importance Resampling (DSIR) | NeurIPS 2023 | selecting domain data | 1 | ✅ |
| E5 | Penedo et al., The FineWeb Datasets | NeurIPS 2024 Datasets & Benchmarks | our pre-training data and how it was filtered | 1 | ✅ |
| E6 | DeepSeek-AI, DeepSeek-R1 (distillation sections) | arXiv 2501.12948; Nature 645 (2025) | distilling into small models | 1 | ✅ |
| E7 | Du et al., Improving Factuality and Reasoning through Multiagent Debate | ICML 2024 (read arXiv v1) | several LLMs reviewing each other | 1 | ✅ |
| E8 | Hinton, Vinyals, Dean, Distilling the Knowledge in a Neural Network (**added**) | NIPS 2014 Deep Learning Workshop | what distillation means | 1 | ✅ |
| E9 | Muennighoff et al., Scaling Data-Constrained Language Models (**added**) | NeurIPS 2023 | repeating shards for several epochs | 2 | ✅ |
| E10 | Gebru et al., Datasheets for Datasets | Communications of the ACM 64(12), 2021 (arXiv v8) | documenting our dataset | 2 | ✅ |
| E11 | Liu et al., G-Eval | EMNLP 2023 | LLM-as-judge (and its limits) | 3 | ✅ |
| E13 | Busbridge et al., Distillation Scaling Laws (**added**, cited by DeepSeek-R1) | arXiv 2502.08606, 2025 | how student size and teacher data trade off | 2 | ✅ |
| E12 | Karpathy, nanochat (code + Discussion #1) | 2025 | reference pipeline on the same data | 2 | ◐ |
| E14 | Allen-Zhu & Li, Physics of Language Models: Part 3.1, Knowledge Storage and Extraction | arXiv 2309.14316, 2023 (ICML 2024) | knowledge is only extractable when each fact appears in many forms in pre-training; "celebrities help minorities" (mixing QA in helps facts seen only in text); why every cat fact is written in six forms and goes into pre-training, not only SFT | 1 | ☐ |
| E15 | Allen-Zhu & Li, Physics of Language Models: Part 3.2, Knowledge Manipulation | arXiv 2309.14402, 2023 | what a model can and cannot do with stored facts (classify, compare, invert) | 2 | ☐ |
| E16 | Berglund et al., The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A" | ICLR 2024 (arXiv 2309.12288) | a fact learned one way is not known the other way; why the knowledge text has a reverse form and a buyer's-thought form | 1 | ☐ |
| E17 | Golovneva et al., Reverse Training to Nurse the Reversal Curse | arXiv 2403.13799, 2024 | training on reversed text as a fix for E16 | 2 | ☐ |
| E18 | Qwen Team, Qwen2.5 Technical Report | arXiv 2412.15115, 2024 | the teacher (Qwen2.5-32B-Instruct, Apache-2.0) that writes the buyer dialogues and knowledge texts | 1 | ☐ |
| E19 | Lin et al., AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration | MLSys 2024 | the 4-bit teacher on Kaggle's T4s; its dequantise path is why long prefills ran out of memory (devlog 2026-10-10) | 3 | ☐ |
| E20 | Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM) | SOSP 2023 | the batched teacher: KV-cache blocks, prefix caching, chunked prefill | 3 | ☐ |

## F. Vision: pretraining, landmarks, cat and drawing datasets

| # | Paper | Venue | Why | Tier | Status |
|---|---|---|---|---|---|
| F1 | Dosovitskiy et al., An Image is Worth 16x16 Words (ViT) | ICLR 2021 | the encoder of the landmark model | 2 | ☐ |
| F2 | He et al., Masked Autoencoders Are Scalable Vision Learners (MAE) | CVPR 2022 | self-supervised pretraining on unlabeled cats; lowered our test NME 0.078 → 0.067 | 1 | ☐ |
| F3 | Xu, Zhang, Zhang & Tao, ViTPose | NeurIPS 2022 | the landmark decoder (classic decoder on ViT features) | 1 | ☐ |
| F4 | Sun, Xiao, Wei, Liang & Wei, Integral Human Pose Regression | ECCV 2018 | soft-argmax: a landmark as the heat map's expected position | 2 | ☐ |
| F5 | Zhang, Sun & Tang, Cat Head Detection: How to Effectively Exploit Shape and Texture Features | ECCV 2008 | the CAT dataset (9 landmarks, 9,996 photos) | 2 | ☐ |
| F6 | Choi, Uh, Yoo & Ha, StarGAN v2 | CVPR 2020 | AFHQ, the 5,558 sharp cat faces the generator learns from | 2 | ☐ |
| F7 | Chan, Durand & Isola, Learning to Generate Line Drawings that Convey Geometry and Semantics (Informative Drawings) | CVPR 2022 | photo → line drawing for the cat faces (`infdraw.py`) | 2 | ☐ |
| F8 | Peng et al., Moment Matching for Multi-Source Domain Adaptation (DomainNet) | ICCV 2019 | the sketch domain used by `pixsketch.py` | 3 | ☐ |
| F9 | Kucer et al., DeepPatent: Large Scale Patent Drawing Recognition and Retrieval | WACV 2022 | a drawing dataset considered, not used | 3 | ☐ |
| F10 | Tian, Clanuwat, Suzuki & Kitamoto, Ukiyo-e Analysis and Creativity with Attribute and Geometry Annotation | arXiv 2106.02267, 2021 | a face dataset considered, not used | 3 | ☐ |

## G. Image generation: tokenizers, guidance, painting with strokes

| # | Paper | Venue | Why | Tier | Status |
|---|---|---|---|---|---|
| G1 | van den Oord, Vinyals & Kavukcuoglu, Neural Discrete Representation Learning (VQ-VAE) | NeurIPS 2017 | image tokenizer (`vqvae.py`) | 2 | ☐ |
| G2 | Razavi, van den Oord & Vinyals, Generating Diverse High-Fidelity Images with VQ-VAE-2 | NeurIPS 2019 | tokenizer details (`vqvae.py`) | 3 | ☐ |
| G3 | Ramesh et al., Zero-Shot Text-to-Image Generation (DALL·E) | ICML 2021 | tokens + GPT for images (`pixsketch.py`, `catface.py`) | 3 | ☐ |
| G4 | Esser, Rombach & Ommer, Taming Transformers for High-Resolution Image Synthesis | CVPR 2021 | VQ-GAN: adversarial loss and adaptive weight for the cat-face tokenizer | 2 | ☐ |
| G5 | Isola, Zhu, Zhou & Efros, Image-to-Image Translation with Conditional Adversarial Networks (pix2pix) | CVPR 2017 | the PatchGAN discriminator | 3 | ☐ |
| G6 | Ho & Salimans, Classifier-Free Diffusion Guidance | NeurIPS 2021 Workshop (arXiv 2207.12598) | classifier-free guidance in the drawing GPT (`drawgpt.py`, `llm/model.py`) | 2 | ☐ |
| G7 | Huang, Heng & Zhou, Learning to Paint with Model-based Deep Reinforcement Learning | ICCV 2019 | painting as an ordered list of strokes; named by the owner for cat generator v2 | 1 | ☐ |
| G8 | Liu et al., Paint Transformer: Feed Forward Neural Painting with Stroke Prediction | ICCV 2021 | the straight oil stroke the owner chose | 1 | ☐ |
| G9 | Zou et al., Stylized Neural Painting | CVPR 2021 | fitting strokes to a photo through a differentiable renderer (`catbrush.py`) | 1 | ☐ |
| G10 | Arar et al., SwiftSketch: A Diffusion Model for Image-to-Vector Sketch Generation | SIGGRAPH 2025 | source of ControlSketch, the 32-Bézier sketches of `linegpt.py` | 2 | ☐ |
| G11 | Du et al., ControlSketch-Part release | 2026, CC BY 4.0 (as cited in `linegpt.py`; title to verify) | part annotations for those sketches | 3 | ☐ |

## H. Cat facts behind the buyers' knowledge

Sources of `ml/sellanything_ml/buyers/cat_knowledge.py`; web articles there are not listed.

| # | Paper | Venue | Why | Tier | Status |
|---|---|---|---|---|---|
| H1 | Quimby et al., 2021 AAHA/AAFP Feline Life Stage Guidelines | JFMS / JAAHA 2021 | the age stages and what each stage needs | 2 | ☐ |
| H2 | AAHA/AAFP Feline Vaccination Guidelines | (cited through secondary summaries; to verify) | kitten vaccine schedule | 3 | ☐ |
| H3 | PKD1 mutation prevalence in cats (PMC6656814) | (title to verify) | polycystic kidney disease in Persians, a held-out topic | 3 | ☐ |

Added = not mentioned in the conversation before, added because the thesis needs them.
