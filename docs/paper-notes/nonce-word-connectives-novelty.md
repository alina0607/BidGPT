# 「小模型像外國學生，靠轉折詞推生字」有沒有人做過？

日期：2026-10-14。性質：文獻查證，不訓練模型、不用 GPU、不 commit。

## 0. 先講結論

| 部分 | 有人做過嗎 | 最接近的論文 |
|---|---|---|
| 小資料量模型的語言能力 | 做過很多 | BabyLM、TinyStories、Chang & Bergen 2022 |
| 從幾句上下文學新詞的意思 | 做過很多 | Lazaridou 2017、Herbelot & Baroni 2017、Schick & Schütze、WinoDict、Minnow |
| 把模型當成第二語言學習者 | 做過，但問的是文法遷移，不是「猜生字」 | Oba 2023、SLABERT 2023 |
| 模型懂不懂轉折詞 | 做過，主要是「預測哪個連接詞」 | Pandia 2021、Beyer 2021、DisSent |
| **用 nonce word + 轉折詞，量模型能不能推出方向** | **做過，而且跟你的想法很像** | **WUGNECTIVES（2025）** |
| 子詞切法把字義切壞 | 有人做 | Hofmann 2021 |
| 打字雜訊與錯字的穩健性 | 做過，但不是在談判或小模型的設定 | Belinkov & Bisk、Pruthi、Moradi & Samwald |

**最重要的一點：WUGNECTIVES 已經做了「用 nonce word 加轉折詞，看模型能不能推出方向」。**
所以你的題目不能再說「用 nonce word 測轉折詞推理」是新的。
它的範圍有三個空缺，是你可能做的，見第 4 節。

誠實的限制：
- 這份只靠網路搜尋。「我沒找到」不等於「不存在」。
- 多數論文我只讀到摘要頁（ACL Anthology、arXiv）。
  只有 WUGNECTIVES 我讀了全文（19 頁 PDF 抽出文字）。
- 摘要頁是用工具抓的，我已盡量用兩個來源對照作者、年份、會議。

## 1. 逐篇查證（都實際打開過原文頁面）

### 1.1 小資料量模型

**BabyLM Challenge**
- Warstadt, Choshen, Mueller, Williams, Wilcox, Zhuang。*Call for Papers: The BabyLM Challenge: Sample-efficient pretraining on a developmentally plausible corpus*。2023，arXiv:2301.11796。
  頁面：https://arxiv.org/abs/2301.11796
- 結果報告：Warstadt 等人（共 11 位作者）。*Findings of the BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora*。CoNLL BabyLM 工作坊，新加坡，2023，頁 1–34。
  頁面：https://aclanthology.org/2023.conll-babylm.1/
- 做法：訓練資料限制在 1000 萬或 1 億個字。評測是文法最小對比（例如 BLiMP）和語言理解任務。
- 跟我們比：它測整體語言能力，不測「讀懂生字」，也不測轉折詞。
  我們的 B3 預訓練是 1.28B token，比 BabyLM 的 1 億字大一個量級，所以不能直接套 BabyLM 的結論。

**TinyStories**
- Eldan & Li。*TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* 2023，arXiv:2305.07759。頁面上沒有列期刊或會議。
  頁面：https://arxiv.org/abs/2305.07759
- 做法：用 GPT-3.5/GPT-4 生成只含 3 到 4 歲小孩會懂的詞的短故事。模型小於 1000 萬參數，仍能寫出文法幾乎正確的段落。
  用 GPT-4 當老師來打分。
- 跟我們比：它證明「詞彙簡單的資料能讓小模型很強」。
  我們的反面問題是：資料詞彙之外的詞，小模型怎麼辦。它沒有測這個。

**Word Acquisition in Neural Language Models**
- Chang & Bergen。TACL 第 10 卷，2022，頁 1–16。
  頁面：https://aclanthology.org/2022.tacl-1.1/
- 做法：追蹤 LSTM、BERT、GPT-2 在訓練中學每個詞的曲線，對照小孩學詞的年齡。
- 發現：模型比小孩更依賴詞頻；訓練早期靠單字頻率，之後才慢慢靠前後文。
- 跟我們比：支持「訓練早期靠頻率」這個背景。
  它沒有測轉折詞或生字推理。
  我們的 ▁hair 後面最常接 less，很像這種頻率偏向，但這只是類比，我們沒有驗證。

**Minnow（BabyLM 規模的學新詞）**
- Wang, Jiang, Linzen, Lake。*Rapid Word Learning Through Meta In-Context Learning*。EMNLP 2025，頁 32038–32073。arXiv:2502.14791。
  頁面：https://aclanthology.org/2025.emnlp-main.1631/
- 做法：用小孩規模的語料從零訓練，用一個占位符代表新詞，訓練模型看幾個例句後生成新用法。
  也拿現成的大模型做同樣的微調。
- 發現（摘要）：從零訓練的小模型，學新詞的能力和用多出幾個數量級資料預訓練的大模型相當。
- 跟我們比：這是最接近「小資料量加學新詞」的一篇。
  它的做法是專門的 meta 訓練，不用轉折詞，也沒有談判情境。
  模型大小我只在摘要頁沒看到，未確認。

### 1.2 從上下文學罕見詞、新詞

**Lazaridou, Marelli, Baroni 2017**
- *Multimodal Word Meaning Induction From Minimal Exposure to Natural Text*。*Cognitive Science* 41(S4)，2017，頁 677–705。
  作者、年份與期刊我用 Crossref API 查到（DOI 10.1111/cogs.12481）。
  Wiley 頁面本身抓取被擋，內容摘要來自搜尋結果。
- 做法：給人和分佈式語意模型看某個新詞的 2 到 6 句自然語句，問它能不能形成基本意思。
- 發現：人和模型都能從這麼少的證據形成基本的意思。
- 跟我們比：這是「從上下文猜生字」的認知科學源頭。
  模型是詞向量，不是會對話的 GPT，也不看轉折詞。

**Herbelot & Baroni 2017（Nonce2Vec）**
- *High-risk learning: acquiring new word vectors from tiny data*。EMNLP 2017，頁 304–309。
  頁面：https://aclanthology.org/D17-1030/
- 做法：改 word2vec，讓新詞的向量從上下文詞的和開始、用高學習率更新。
- 跟我們比：同樣是詞向量時代，沒有轉折詞。

**Schick & Schütze（兩篇）**
- *Attentive Mimicking: Better Word Embeddings by Attending to Informative Contexts*。NAACL 2019，頁 489–494。
  頁面：https://aclanthology.org/N19-1048/
- *Rare Words: A Major Problem for Contextualized Embeddings and How to Fix it by Attentive Mimicking*。AAAI 2020（arXiv:1904.06707）。
  頁面：https://arxiv.org/abs/1904.06707
- 做法：用字形和所有上下文，給罕見詞估計更好的向量，對 BERT 加上「單一 token 近似」。
- 發現：BERT 即使預訓練資料很大，處理罕見詞還是不好。
- 跟我們比：這確認「大模型也怕生字」，但解法是改嵌入，不是讓模型自己靠轉折詞推。
  AAAI 2020 的頁面我只看到「To appear at AAAI 2020」，會議資訊靠 dblp 與搜尋結果補。

**WinoDict**
- Eisenschlos, Cole, Liu, Cohen。*WinoDict: Probing language models for in-context word acquisition*。EACL 2023，頁 94–102。
  頁面：https://aclanthology.org/2023.eacl-main.7/
- 做法：把 Winograd 題目裡的關鍵詞換成編造的詞，在提示裡附上字典定義。
- 發現：換成新詞後準確率大幅下降。
- 跟我們比：它給「定義」，我們要模型靠「上下文暗示」。

**CoLLEGe**
- Teehan, Lake, Ren。*CoLLEGe: Concept Embedding Generation for Large Language Models*。arXiv:2403.15362，2024（搜尋結果說是 COLM 2024，arXiv 頁面本身沒列）。
  頁面：https://arxiv.org/abs/2403.15362
- 做法：用元學習從幾個例句產生新概念的嵌入。評測包含 GRE 填空和網路俚語。
- 跟我們比：和 Schick 同類，不用轉折詞。

**收集清單**：Kanishka Misra 整理的 `wugs-and-daxes`（29 篇新詞論文）：https://github.com/kanishkamisra/wugs-and-daxes
想找更多可以從這裡找。

### 1.3 第二語言習得

**Oba 等人 2023**
- Oba, Kuribayashi, Ouchi, Watanabe。*Second Language Acquisition of Neural Language Models*。Findings of ACL 2023，頁 13557–13572。
  頁面：https://aclanthology.org/2023.findings-acl.856/
- 做法：先用第一語言預訓練，再用第二語言訓練，看文法能不能學得更快。
- 發現：第一語言預訓練讓第二語言的文法歸納更快；第一語言的選擇、有沒有平行文本都有影響。
- 跟我們比：它問的是「文法遷移」，不是「猜不認識的詞」。
  模型大小我在頁面沒看到。（另一篇文章說是 18M 參數，我沒有在原文確認，不採用。）

**SLABERT**
- Yadavalli, Yadavalli, Tobin。*SLABERT Talk Pretty One Day: Modeling Second Language Acquisition with BERT*。ACL 2023（長論文），頁 11763–11777。arXiv:2305.19589。
  頁面：https://aclanthology.org/2023.acl-long.657/
- 做法：用 MAO-CHILDES（五種語言的兒童導向語料）當第一語言，再學英文，用 BLiMP 評測。
- 發現：語言家族距離越遠，負遷移越多；對話語料比腳本語料更有幫助。
- 跟我們比：同樣是文法遷移，不測生字。

**Constantinescu 等人 2025（只用來補脈絡）**
- Constantinescu, Pimentel, Cotterell, Warstadt。*Investigating Critical Period Effects in Language Acquisition through Neural Language Models*。TACL 2025。
  作者與期刊我用 Crossref 查到（DOI 10.1162/tacl_a_00725），沒有讀內容。

### 1.4 語言模型與轉折詞

**Pandia, Cong, Ettinger 2021**
- *Pragmatic competence of pre-trained language models through the lens of discourse connectives*。CoNLL 2021，頁 367–379。
  頁面：https://aclanthology.org/2021.conll-1.29/
- 做法：用填空任務，測模型能不能用語用線索預測連接詞，懂不懂連接詞帶的隱含意思。
- 發現：在自然文本上預測得不錯，但在控制過的語境下大幅下降。
  作者的結論是目前主流預訓練做法不會讓模型有足夠的語用能力。
- 跟我們比：測的是「預測連接詞」，不是「用連接詞推生字」。
  模型名稱與大小在摘要頁沒有寫，我沒有看全文。

**Beyer, Loáiciga, Schlangen 2021（CoherenceGym）**
- *Is Incoherence Surprising? Targeted Evaluation of Coherence Prediction from Language Models*。NAACL 2021，頁 4164–4173。
  頁面：https://aclanthology.org/2021.naacl-main.328/
- 做法：做一組針對連貫性的測試，不只是打亂句子順序。
- 跟我們比：同樣是「模型看得出不連貫嗎」，不是推生字。

**DisSent**
- Nie, Bennett, Goodman。*DisSent: Learning Sentence Representations from Explicit Discourse Relations*。ACL 2019，頁 4497–4510。
  頁面：https://aclanthology.org/P19-1442/
- 做法：用明確的連接詞自動建大量訓練資料，讓模型學預測兩句之間的連接詞，得到句子向量。
- 跟我們比：這是「把連接詞當訓練訊號」的做法，和我們「靠預訓練與 SFT 讓模型自己學會」同方向。
  但它用 BiLSTM，目標是句子嵌入，不是生字推理。

**WUGNECTIVES（最重要的一篇）**
- Brubaker, Sheffield, Li, Misra（德州大學奧斯汀分校語言系）。*WUGNECTIVES: Novel Entity Inferences of Language Models from Discourse Connectives*。arXiv:2510.09556，2025-10 首次提交，v3 為 2026-02-01。arXiv 頁面沒列正式會議。
  頁面：https://arxiv.org/abs/2510.09556
- 做法（我讀了全文）：
  - 用 nonce word（wugs、daxes、feps 等）當新實體，前提句用轉折詞連起兩個新詞，例如 "Although I hate leafy vegetables, I prefer wugs to daxes"。
    問「Wugs are leafy vegetables?」，答案是是/否，所以隨機基線是 50%。
  - 共 740 句，套 12 種提示變化，得到 8,880 個刺激。
  - 41 個連接詞形式，7 種語意，包含 although、but、however、yet、even though、though、because、so、therefore 等。
  - 測 17 個開源模型：Qwen-2.5（0.5B 到 14B，base 與 instruct 共 10 個）、OLMo-2（1B 與 7B，4 個）、Llama 3.1 8B（2 個）、再加一個用 DeepSeek R1 推理過程微調過的 Qwen2.5-14B。
- 發現：
  - 時間關係與因果（because）高於隨機，但**讓步（although、but、however 等）所有模型都接近隨機**。
  - 沒有明顯的模型大小或指令微調的效果；用推理過程微調有幫助，但讓步仍然很差。
  - 論文說這不能用連接詞在網路語料的頻率解釋。
  - 換成真實實體（有世界知識）準確率明顯上升，但讓步上升最少。
- 跟我們比：
  - **這幾乎就是你的想法**：nonce word 加轉折詞推方向，連 although 和 because 對調方向的設計都有。
  - 它測的是 0.5B 以上的現成模型，**沒有**自己控制預訓練量或 SFT 資料。
  - 它用是/否的蘊含判斷，**沒有**談判、出價這種行為結果，也沒有錯字。
  - 它沒有 71M 這個尺度的模型。

### 1.5 子詞切法與錯字穩健性

**Superbizarre Is Not Superb**
- Hofmann, Pierrehumbert, Schütze。*Superbizarre Is Not Superb: Derivational Morphology Improves BERT's Interpretation of Complex Words*。ACL-IJCNLP 2021，頁 3594–3608。
  頁面：https://aclanthology.org/2021.acl-long.279/
- 做法：發現 BERT 的 WordPiece 常常把複雜詞切成沒有意義的片段，改用詞綴切分的 DelBERT 效果明顯較好（搜尋摘要說對少見的複雜詞差距最大，我沒在 ACL 摘要頁核對）。
- 跟我們比：**這和我們看到的 long-haired 被切成 long - ha ired 同一類現象。**
  它支持「切法不對會把字義切壞」，但它是 BERT、沒有轉折詞，也沒有談判。

**Belinkov & Bisk**
- *Synthetic and Natural Noise Both Break Neural Machine Translation*。ICLR 2018，arXiv:1711.02173。
  頁面：https://arxiv.org/abs/1711.02173
- 發現：字元層級的翻譯模型遇到自然和人造雜訊都很脆弱。

**Pruthi, Dhingra, Lipton**
- *Combating Adversarial Misspellings with Robust Word Recognition*。ACL 2019，頁 5582–5591。
  頁面：https://aclanthology.org/P19-1561/
- 發現：對 BERT 情感分類器，一個最壞情況的字元改動就能把準確率從 90.3% 降到 45.8%；先放一個詞辨識器可以回到 75%。

**Moradi & Samwald**
- *Evaluating the Robustness of Neural Language Models to Input Perturbations*。EMNLP 2021，頁 1558–1570。
  頁面：https://aclanthology.org/2021.emnlp-main.117/
- 發現：BERT、XLNet、RoBERTa、ELMo 遇到小的字元或詞層級擾動都會變差。

**談判加錯字**：我搜尋「談判遊戲加拼字錯誤」沒有找到論文。
最接近的是 2026 年的多屬性議價研究（我只讀到搜尋摘要，沒開原文），它沒有測雜訊。

### 1.6 其他我只看到列表、沒有打開原文的

下面這些我只在搜尋結果或別的論文的引用裡看到，**沒有核對原文頁面，筆記裡不依賴它們**：
- Landauer & Dumais 1997，*A Solution to Plato's Problem*，*Psychological Review*：這是 TOEFL 同義詞測驗的源頭。作者與期刊我用 Crossref 確認。
  搜尋結果寫 LSA 在 TOEFL 同義題的正確率約 64%，與非母語大學申請者的平均相近，這個數字來自二手整理，使用前要看原文。
- Kobayashi, Okazaki, Inui 2017，IJCNLP（頁面 I17-1048）：RNN 語言模型動態表示未知詞。
- Miao 等人 2024（DISQ）、Cong 等人 2023：只在 WUGNECTIVES 的相關研究裡看到。
- 一篇談連接詞對人類與模型預測影響的 PMC 論文（PMC11471541）：頁面被擋，沒有讀到。
- Bunzeck 等人 2025 *Small Language Models Also Work With Small Vocabularies*（COLING 2025，arXiv:2410.01487）：我讀了摘要頁，說小型 Llama 用字元詞彙也能學好；跟「子詞切法對小模型的影響」有關，可作之後的參考。

## 2. 跟我們題目的比較表

| 論文 | 測什麼 | 模型與資料大小 | 用轉折詞推方向嗎 |
|---|---|---|---|
| BabyLM | 整體文法與理解 | 1000 萬、1 億字 | 否 |
| TinyStories | 小模型的流暢度 | 少於 1000 萬參數 | 否 |
| Chang & Bergen | 詞的學習曲線 | LSTM、BERT、GPT-2 | 否 |
| Minnow | 看幾句學新詞 | 小孩規模語料從零訓練 | 否 |
| Lazaridou 2017 等 | 幾句上下文學新詞 | 詞向量 | 否 |
| Schick & Schütze | 罕見詞嵌入 | BERT | 否 |
| WinoDict | 給定義學新詞 | 大模型 | 否 |
| Oba / SLABERT | 第二語言的文法遷移 | 摘要頁沒寫大小 | 否 |
| Pandia / Beyer / DisSent | 預測或偵測連接詞 | 預訓練模型 | 是，但測預測連接詞 |
| **WUGNECTIVES** | **nonce word 加連接詞推屬性** | **0.5B 到 14B 現成模型** | **是** |
| Hofmann 2021 | 子詞切法與複雜詞 | BERT | 否 |
| 錯字穩健性 | 打字雜訊 | 各種 | 否 |

## 3. 已經有人做的

1. 「小資料量模型能學到多少語言」：BabyLM、TinyStories。
2. 「從幾句上下文學新詞」：Lazaridou、Herbelot、Schick、Minnow、CoLLEGe。
3. 「模型懂不懂連接詞」：Pandia、Beyer、DisSent。
4. **「nonce word 加連接詞，推屬性方向」：WUGNECTIVES。**
   它已經發現讓步詞（although、but、however）所有模型都接近隨機，這和你的「轉折詞」重點直接相關。
5. 「子詞切法壞會讓字義壞」：Hofmann 2021。
6. 「錯字讓模型變差」：一系列 NLP 穩健性論文。

## 4. 可能是新的角度（只代表我沒找到，不代表一定沒人做）

1. **在 71M 這個尺度，控制預訓練量與 SFT 資料，看轉折詞推理怎麼變。**
   WUGNECTIVES 用的是 0.5B 以上現成模型，沒有自己控制訓練資料，所以不能回答「這個能力是預訓練給的還是 SFT 給的」。
   我們可以固定模型、只改預訓練 token 數（例如 0.1B、0.4B、1.28B）和 SFT 的場數（例如 80、160、320 場）。
2. **行為結果：出價方向，而不是是/否判斷。**
   我搜尋談判加 nonce word 或轉折詞沒有找到論文。
   做法是把屬性詞換成 nonce word，轉折詞前後調換，看買家的出價是不是往預期方向動，並跟 WUGNECTIVES 的是/否題對照。
3. **子詞切法與轉折詞暗示的衝突。**
   我們實際看到 long-haired 被切成 long - ha ired，▁hair 後面最常接 less。
   可以做一個受控實驗：同一個生字有「切法讓前綴頻率偏向相反意思」與「切法中性」兩種，看轉折詞的暗示能不能蓋過子詞頻率的偏向。
   Hofmann 做了切法的影響，WUGNECTIVES 做了連接詞，兩者都沒有放在一起。
4. **錯字與生字的交互作用。**
   使用者的錯字本質上也是生字。可以把錯字和 nonce word 放在同一個量表上比較。

## 5. 做之前要注意

- 要先讀 WUGNECTIVES 的 GitHub 資料（https://github.com/kanishkamisra/wugnectives），
  考慮直接拿它的刺激當評測集，不要重做。這樣就能和 17 個模型的數字對照。
- WUGNECTIVES 的 nonce word 在英文裡是整個被切成幾片。我們的 sp8192 會把 wug、dax 切成什麼，要先量。
  nonce word 的子詞切法本身也會影響結果。
- 專案規矩：模型只能靠預訓練加 SFT 變好。這份題目是**評測**，不碰規則或補丁。
  如果結果顯示切法是原因，可以改的是 tokenizer 和資料，這在規矩允許範圍內。
  不能做的是為了這個失敗加一條特例。
- 我們目前對 hairless 的解釋（子詞切法加上詞頻偏向）是**從觀察猜的，沒有做受控實驗**。
  不要在報告裡寫成已證實。
- 還沒有確認的事：WUGNECTIVES 是否已被某個會議接受；Minnow 的模型大小；Pandia 用了哪些模型。


## 附錄 A：embedding 介入診斷實驗設計（只是設計，還沒寫程式、沒跑）

日期：2026-10-14。

### A.1 為什麼做、做了不做什麼

問題：B3 讀錯「long hair」或「pure breed but no paper」，是因為某個詞的表示不對，還是更深的地方出錯？
做法：在推論時，把某個詞的輸入向量換掉，看輸出有沒有變。這是**診斷**，不是功能。

規矩：
- 結果**不會**放進遊戲，也不會變成「遇到 but 就改向量」的規則（那是補丁）。
- 如果診斷顯示問題在詞的表示，之後只能改資料、分詞器或預訓練量，再重新訓練。
- 不訓練、不用 GPU（71M 在 CPU 上推論；`tools/probe_misread.py` 的說明也是在 CPU 上取樣）。

### A.2 先確認的程式事實（讀過 `ml/sellanything_ml/llm/model.py`）

- 輸入和輸出共用同一張 embedding 表（第 190 行 `self.head.weight = self.embed.weight`）。
  所以如果直接改表裡的一列，會同時改輸入和輸出。
  **介入要用 forward hook，在 `embed` 的輸出上換指定位置的向量**，這樣只動輸入，不動輸出表。
- 預設 `d_model = 512`、`vocab_size = 8192`。
- 專案裡已經有 `start_rows_from_words`（第 309 行）：新 token 的向量用它的字的各列平均當起點，之後會在 SFT 中學。
  I2 這個條件借用同樣的算法，但這裡**不接訓練**。
- 可重用的現成工具：`tools/probe_list_tokens.py`（先列出分詞結果，我還沒讀它的內容，要先確認）、
  `tools/probe_misread.py`（取樣方式與 `eval_buyer_b1.py` 相同）。

### A.3 測試案例

從已看到的失敗開始，每個案例做成一組最小對比：
1. "he has a long hair" 被讀成 hairless。
2. "intact" 不認得。
3. "pure breed but no paper" 被讀成沒證書也不是純種。

做之前先用 `probe_list_tokens.py` 列出每個詞被 sp8192 切成什麼，記下來。
案例要多到能看出趨勢，我建議先做約 20 組，並且每組都附一個「換成熟詞」的對照句。
題目取自評測用資料（不能是訓練裡出現過的句子）。

### A.4 條件

| 條件 | 做法 | 目的 |
|---|---|---|
| 基準 | 不介入 | 現在的錯誤率 |
| I1 熟詞上限 | 把這個詞所有片段的輸入向量換成一個熟悉同義詞的向量 | 如果模型「認識這個意思」，應該修好。這是上限 |
| I2 片段平均 | 換成這個詞各片段向量的平均 | 檢查是不是切碎造成的 |
| I3 上下文平均 | 換成同一句其他字的平均 | Nonce2Vec 式的起點 |
| I4 負對照 | 換成同範數的隨機向量，和換成不相關詞的向量 | 看輸出變化是不是只是雜訊 |
| I5 反向 | 換成原向量乘 −1 | 驗證「反方向」這個假設（我預期會亂掉，要實測） |

每個介入都要記下：換進去的向量範數 vs 表裡一般列的範數。範數差太多，結果就可能只是「模型沒見過這種輸入」。

### A.5 指標

- 優先用**確定性**的量：比較「正確讀法」與「錯誤讀法」的續寫對數機率差，不靠取樣。
- 再補取樣版：沿用 `probe_misread.py` 的方式（溫度 0.7、top-k 50、固定種子），看讀錯率。
- 談判的行為指標：出價方向對不對。這跟 N1 講義後的 B22 構想相連。

### A.6 怎麼解讀

| 結果 | 結論 | 之後 |
|---|---|---|
| I1 修好，I2/I3 沒有 | 模型懂意思，問題在生字的表示 | 改資料覆蓋、分詞器或預訓練量，重新訓練 |
| I1 也沒修好 | 問題在更深的層，例如讀 but 與否定 | 看讓步句的訓練資料，不是 embedding |
| I4 的變化和 I1 一樣大 | 輸出對輸入向量本來就不穩，結果是雜訊 | 這個診斷不能下結論 |
| I5 讓方向合理翻轉 | 出乎預期，要多量幾個案例再相信 | 不採用，只記錄 |

### A.7 限制

- 只換一個位置的向量，不能證明模型內部真的「靠它想」。這是相關性，不是機制證明。
- 換進去的向量不在模型見過的分布裡，可能直接壞掉。所以才要 I4 和範數檢查。
- 20 組太少，不夠寫成論文裡的結論，只夠決定要不要往下做。
- 這份設計還沒有任何程式，也沒有跑過；所有條件的預期結果都是我的推測，不是實測。
