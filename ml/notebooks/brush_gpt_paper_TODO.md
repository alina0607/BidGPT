# Brush GPT 論文:待辦清單(TODO)

> 對象:`brush_gpt_paper.ipynb`。
> 策略:**現在先給教授看 notebook 版**,但寫法盡量照論文標準,之後轉 Overleaf 時只需要「搬」,不必「重寫」。
> 每一項標了它屬於哪個階段:
> **[現在]** = 給教授看之前就該處理;**[轉論文]** = 轉 Overleaf 時再處理(現在先別動,免得教授版讀起來缺東西)。

---

## ✅ 已完成(2026-10-18)

數字與公式全部對過程式碼(`catbrush.py`、`brushtokens.py`、`brushgpt.py`、cat-face-landmarks 的
`mae.py`、`pose.py`、`label.py`、`finetune.py`、`landmarks.py`),修正如下:

| 位置 | 問題 | 怎麼改 |
|---|---|---|
| §1.2、§4.2、§4.3.3、Figure 8c/8d 圖說 | 3,046 和 3,064 兩個數字都出現,但沒說差在哪,讀起來像打錯字 | 兩個都對:**3,046 = 沒有臉部點**(trial、A–D、F、H);**3,064 = 有臉部點**(E、G、I、J)。每一處都標清楚是哪一種 |
| §4.2 版面公式 | 角度寫成 $\theta_i$,但 §4.1.2 寫 $\phi_i$,§5.1 的 $p_\theta$ 又把 θ 當模型參數 | 統一成 $\phi_i$ |
| §4.1.3 疊色公式 | `clip(c_i S_i)` 少了 `0, 1`(§4.1.2 有寫);$\prod_{j>i}$ 沒有上限 | 補成 `clip(c_i S_i, 0, 1)` 和 $\prod_{j>i}^{n_k}$ |
| §3.2 MAE 公式 | 沒說 var 是哪一種 | 註明是不偏樣本變異數(除以 767),跟程式的 `torch.var` 一致 |
| §3.4 | 規則排除後剩 6,830 隻,但 Table 3 的 CAT 是 6,848 隻,差 18 | 補一句:6,983 隻畫完的貓裡有 18 隻**沒有眼睛分數**,規則檢查不到,所以留在資料裡 → 6,983 − 135 = 6,848 |

核對過、**沒有錯**的地方:J 那一格「Full model scores 3.933」是 I 在 AFHQ val(3.9202,292 隻)和
CAT val(3.9407,521 隻)上按貓數加權的結果(3.9334),跟 D 的 3.9331 只是巧合。

---

## ✅ 已完成(2026-10-18,第二輪)

- **Table 2d 的標籤**:`kept` → `kept (of the paintings with eye measures)`。
  ⚠️ 只改了標籤文字(程式碼和存下來的輸出都改了,數字沒動),**還沒在 Mac 上重跑**。
- **`catbrush.py` 的 docstring**:三處「along the fur」改成「along the photo's edges」,跟論文 §4.1.4 一致(程式行為沒變)。
- **正文的開發日誌痕跡**:日期、「development log」、「project owner's agreement」、「hand-off note 7,531」都清掉了。
  被拿掉的出處**沒有丟**,記在下面「Appendix E 要列的出處」,寫附錄時照抄。
- **18 隻沒有眼睛分數的 CAT 畫作**:新增 `ml/tools/backfill_cat_eye_checks.py` 補算(見下面「要在 Mac 上跑的」)。
  從程式碼推斷的原因:`paint_cat_dataset.py` 從 10-12 03:58 開始畫,眼睛檢查是 10-12 才決定加的,
  **最早畫的那批紀錄裡沒有 `eye_conf`**;而 `brushgpt._painted()` 只有在 `"eye_conf" in e` 時才套規則,所以這批直接通過。
  ⚠️ 這是推斷,沒有資料無法證實;跑腳本時它會把那幾隻列出來,可以對照它們的畫圖時間。

## ✅ 已在 Mac 上完成(2026-10-17)

- `backfill_cat_eye_checks.py` 列出**剛好 18 隻**,`--write` 補算後改寫 `log.jsonl`(舊檔在 `log.jsonl.bak`)。
  只有這 18 行改變,而且只多了四個眼睛欄位,其餘內容不變。
- **推斷已證實**:這 18 張正是 log 最前面的畫作(第 0–18 行),第一行有 `eye_conf` 的是第 20 行,之後沒有任何畫作缺少。
- **18 隻全部通過規則**:排除數仍是 135(1.9%),保留 6,983 − 135 = **6,848**,正好等於 Table 3 的 CAT 貓數。
  Table 2c 的百分位、Figure 6 的數字(比值中位數 0.995、eye_err 0.038 / 0.104)與 Figure 7 的六隻貓都沒有變。
- 因此 **J 的資料(補算前 tokenise)與規則現在保留的貓完全相同**,§3.4 已照此改寫;Table 2d 的標籤改回 `kept`。

## 📎 Appendix E 要列的出處(從正文拿掉的日期與紀錄)

| 正文位置 | 原本寫的出處 |
|---|---|
| §3.2「與事前預期相反的結果」 | development log 2026-10-10 §1 |
| §3.3 重現性說明(Table 2 的 AUROC) | development log;2026-10-10 用未 commit 的臨時腳本算的 |
| Table 4a「Face box」 | decision 2026-10-10(臉部 vs 整張照片) |
| Table 4a「Brush shape」 | decision 2026-10-09(五種筆刷比較;程式在私人 Kaggle dataset `alinachuang/cat-brush-proto`) |
| §4.1.1 只畫臉 | 2026-10-10 決定 |
| §4.2 五 token 版面 | 2026-10-09 與 painter 一起決定 |
| §4.2 詞彙表 | 早期交接筆記寫 7,531(錯),程式為 7,499;development log 2026-10-11 記錄更正 |
| Table 6a、Table 6c 的 Trial 欄 | development log 2026-10-11 §2 |
| §4.3.3 field restriction | 2026-10-11 由專案擁有者決定採用(「這是架構改變還是補丁?」);正文改成指向 Appendix A |

## 🔴 優先處理
- [ ] **[轉論文] 文字的作者**:會議規定可以用 AI 潤稿,**不能由 AI 代寫段落和論點**。
      轉 Overleaf 時每一段用自己的話重寫,Claude 只負責審稿。
- [ ] **[轉論文] 「(made by Claude)」**:給教授看的版本保留(誠實揭露)。
      轉論文時從 §3.2、§4.1 標題拿掉,改寫進 **Acknowledgments / AI-use statement**,並查目標會議的 LLM policy。

## 🟠 範圍與實驗設計(審稿人一定會問)

- [ ] **RQ4(毛流)、RQ5(條件生成)還沒做完**:決定這篇只留 RQ1–RQ3,RQ4/RQ5 移到 Future Work、之後另寫一篇?
      (影響:摘要第 4 段、§1.3、§12、§13、Table 0 最後兩列。)
- [ ] **Judge 不夠獨立**:同一個 landmark model 同時負責標點 → 對齊(H)、定 eye-weight 區域(F)、過濾 CAT 資料,**又當評審**。
      方案:換一個不同架構或不同資料訓練的第二評審;至少在 Limitations 正面寫出來。
- [ ] **FID 樣本太少**(128/256 張,FID 通常要上萬張):小樣本時改以 KID 為主,或補更多樣本。
- [ ] **FID 的比較對象**:Table 0 是跟 *training* paintings 比;主結果應該改用 held-out **test**(摘要的 196 vs 49)。
- [ ] **每個版本只跑一個 seed**:「E、F 各自都在雜訊內,合起來才超過」需要多個 seed 才站得住。
- [ ] **沒有外部 baseline**(例如同資料的 pixel 模型、Sketch-RNN 式的連續輸出):投會議幾乎一定會被要求。
- [ ] **順序控制組**(§4.3.4 自己寫的):原順序 + field restriction 那一組還沒訓練。

## 🧪 眼睛的實驗(2026-10-18 起)

- [x] **逐筆診斷**(`tools/eye_stroke_attribution.py`):疊到眼睛上的筆在第 3、4 階段幾乎不存在(第 4 階段 0 筆);
      真正的差別是有眼睛加權的模型往眼睛區域多畫約一倍的筆(`tools/eye_zone_strokes.py`:真畫 85.1、I 170.6、J 166.1 筆/幅;
      D 對 F:74 → 134,只差眼睛加權),眼睛分數卻幾乎不變。
- [x] **K:眼睛加權不加在位置上**(Kaggle T4 訓練,I 的批次 8×2;筆電評估 `EVAL_ONLY=1 tools/run_brushgpt_K_mac.sh`):
      機制成立(眼睛區域 87.7 筆 vs 真畫 85.1、I 170.6),驗證損失 3.915 → 3.883,
      但眼睛**變差**:溫度 1 時 −0.0235(−2.8 標準誤,判準「hurt」),FID(對測試)196 → 206。
      → 多畫的眼睛筆在補償資料中不夠清楚的眼睛;問題在資料端。已寫進中文 notebook 第 14.2 節。
      還要:英文版 §6.3 補「加權位置的副作用與取捨」。
- [ ] **依模型自己的損失動態調整眼睛權重**(owner 2026-10-18):例如眼睛筆的損失相對偏高時暫時加重、降下來再放鬆。
      **前提**:只加在已經知道是不是眼睛的 token(角度、大小、顏色),不加在位置,否則又會讓模型多畫眼睛。
      **禁止**:用眼睛評審的分數來調權重(評審是評分用的獨立裁判,拿來訓練就不能再當證據)。
      **時機**:K 的結果是「有害」(上面),所以先擱置;等高光層的資料出來後再看。
- [x] **眼睛裁切的畫家比較**(`tools/eye_crop_styles.py`、`tools/eye_crop_figure.py`;中文 notebook §14.3):
      24 隻做完:外框「先描再擬合」比「隨機擬合」好(眼框 +0.0038,20/24;顏色也較好),最後的高光層有效(25 → 31–32 / 35 個高光)。
- [~] **高光作為最後一個階段**(單一模型的作法,§14.4):owner 2026-10-18 改走眼睛模型(下面),保留為替代方案。
- [ ] **資料重新整理(owner 2026-10-18;中文 notebook §15)**:
      - [x] 遮罩稽核 `tools/mask_audit.py`:臉被切掉 >3% 的 AFHQ 10 / CAT 69(籠子、植物、兩隻貓合成一個遮罩)。
      - [x] 分工:整體模型(只學頭與兩耳完整的)+ 三個部位模型(眼睛、耳朵、鼻子嘴巴);毛流是第四個還是分進三個,未定。規則寫在 CLAUDE.md。
      - [x] 五個人工審核資料夾 `tools/review_export.py` → `ml/data/processed/review/`(1 整體打底、2 耳朵、3 眼睛、4 鼻子嘴巴、5 毛流)。
      - [x] 只留頭的遮罩 `review_export.head_shape`(去掉耳後的背、胸口、旁邊的貓)。
      - [ ] **人工審核中**;審完跑 `--collect` 寫 `assignments.csv`(每張照片用在哪些訓練)。
      - [ ] **待實作:不確定區域不學(遮罩損失)**:三種像素——貓 / 背景(真輪廓)/ 不確定(Apple 說貓但在頭形外、照片外;**下巴也算不確定**,owner 決定)。
            畫家:不確定像素權重 0、不在此起筆;資料:三值圖跟筆觸一起對齊/鏡像/重排;訓練:中心在不確定區的筆 5 個 token 權重 0(與眼睛加權相乘);
            驗證損失同樣加權並報告被遮去的筆比例;評估的真畫參考值同樣處理。審核完後一次重畫全部的貓並重訓整體模型。
- [ ] **數位重畫(owner 2026-10-18;中文 notebook §16)**:第一版只畫 AFHQ 5,547 隻,畫完先訓練各種 transformer 看結果,好再補 CAT 9,714 隻。
      - [x] 輸入 `tools/repaint_prep.py` → `data/processed/repaint/inputs/`(15,261 隻、31 批;256 px 目標圖 + 三值圖 + 臉部點 + 裁切框;第 0–10 批與第 11 批前 47 隻是 AFHQ)。
      - [x] 重畫 `tools/repaint.py`:8 層全部存下(原始參數與像素值、每層筆刷設定);不確定區域權重 0;可接續;3 隻驗證過
            重播一致(差 ≤ 0.002 = uint8 誤差)、只畫第 1–5 層、每筆標部位、換算到放大畫布。
      - [~] 進行中:Kaggle 兩張 T4(`alinachuang/brush-repaint-afhq-t4`)GPU0 畫第 0–4 批、GPU1 第 5–9 批,各 680 分鐘後自動停;
            Mac(`tools/run_repaint_mac.sh` → `data/processed/repaint/mac/`)先畫第 10–11 批的 AFHQ,再從第 9、4 批尾巴往回畫。
      - [ ] Kaggle 結束後下載 `out/`、合併(同一隻畫兩次保留一份)、剩下的再分配。
      - [ ] 之後:暈染層在放大畫布上要重算(`render_zoomed` 目前略過暈染);部位畫家(耳朵、鼻嘴)與其計時。
- [ ] **可變長度與構圖草稿(owner 2026-10-19 決定;待實作)**:
      - 每層筆數由模型**邊畫邊決定**:每一筆開始處可選「再畫一筆」或「這層結束」;欄位限制改為依序列狀態推算。**生成時不設下限**(只設與現在相同的上限),改為如實報告生成畫每層、每區的筆數分布並與真畫比較。
      - 「夠了」**每層依它的用途判斷**:打底與色塊看模糊後的大面積;小色塊看中等尺度(條紋寬);暈染看是否還有硬邊;細毛看紋理;最細的深/亮毛看照片細線的覆蓋率。門檻是畫風旋鈕,先用 6 隻 × 2–3 個門檻給 owner 挑。
      - **細節層按部位逐區收尾**(避免模型把整層的筆耗在同一部位;實驗 K 已證明加權偏向某區會多畫一倍):打底四層整張臉一起、每層一個結束 token;細毛與最細毛每層再分區,順序固定為耳朵外側 → 額頭 → 左臉頰 → 右臉頰 → 下巴與嘴周圍,每區一個結束 token(讓模型自己挑下一區之後再當對照實驗)。眼眶、耳內、鼻嘴的細節歸部位模型。
      - 不確定區域:細筆刪除,大筆保留但遮罩損失。
      - **構圖草稿**:臉的角度(用平均貓頭立體模型對齊 9 點,得 yaw/pitch/roll)+ 輪廓(頭形與耳朵取自三值圖的貓/背景交界,下巴不畫;眼眶取 SAM;瞳孔;鼻、嘴)。A:序列開頭的計畫 token(模型自己生成,訓練時以資料為答案);B:展示與作畫動畫時渲染成淡草稿線。兩者都做。
      - **不需重畫**:以「整理步驟」從數位重畫的資料做出(分層分區判斷夠了 → 刪筆 → 不夠的區補筆 → 依區排序 → 短暫再擬合);實測:只刪不再擬合誤差 +0.006,所以必須再擬合。若整理後與「畫家原生分區」差太多,才改畫家重畫。
      - 實測(60 隻):打底層每隻都用滿;細毛 164–369/400、最細深毛 16–240/500、最細亮毛 2–178/500;合計需要 649–1,230/1,848(中位數 873),虎斑約純色貓兩倍。
- [ ] **眼睛模型(owner 2026-10-18 決定,條件寫在 CLAUDE.md)**:另一個模型在放大的眼睛畫布上,於臉部畫作之上畫四層:
      外框 → 虹膜 → 瞳孔 → 高光,顏色不准越過外框。畫家 `tools/eye_layers.py` 已試 24 隻(§14.5):
      相對只有臉部畫作,24/24 隻眼框、眼睛顏色、溢出都變好,高光 10 → 29 / 35。
      下一步:① 描眼框加形狀約束(貓 000041 描壞);② 量瞳孔大小當計畫 token;③ 全部貓做四層資料 + 找更多高畫質貓眼;
      ④ 設計眼睛模型的序列與條件;⑤ 判準先寫死,永遠同時報告「只有臉部模型」與「臉部 + 眼睛模型」。

## 🟡 參考文獻

- [ ] **[5] 和 [44] 是同一篇**(AttentionPainter)→ 合併,並更新正文的引用編號。
- [ ] [44]、[55]、[56]、[57] 用了**全形括號「(」**→ 改成半形(LaTeX 會出問題)。
- [ ] [11]「Wen et al.」缺作者名字縮寫。
- [ ] **每一筆都從 arXiv / DBLP 抓 BibTeX 重新核對**,尤其 2026 年的 arXiv 編號(2603.*、2605.*、2601.*)。
- [ ] §1–§4 沒有引用到 [22]–[31]、[39]、[44]–[52]、[55]–[59];等 §5 之後寫完再檢查一次,沒用到的刪掉。

## 🟡 notebook 痕跡(轉論文時移到附錄)

- [ ] **[轉論文] commit `d6e00d0`、檔名 `catbrush.py` / `modal_brushgpt.py` / `brushgpt.py`**:移到 Reproducibility 附錄。
- [ ] **[轉論文] 摘要裡的「Evidence rule … this notebook」**:改成一個附錄(Appendix E 本來就有)。
- [ ] **[轉論文] GitHub 連結 `alina0607/cat-face-landmarks`**:投雙盲會議時會洩漏身分,要匿名化;arXiv 可以保留。

## 🔵 轉 Overleaf 的結構(轉論文時)

- [ ] 模板:NeurIPS `preprint` 模式(單欄、顯示作者、不限頁數),之後要投別的會議只需要換 `main.tex`。
- [ ] 檔案:`main.tex` + `sections/*.tex`(一節一檔)+ `figures/*.pdf` + `refs.bib`。
- [ ] notebook 每張圖加一行 `plt.savefig("figures/xxx.pdf")` 輸出向量圖。
- [ ] 從「時間順序」改成「邏輯順序」:

| notebook | 論文去處 |
|---|---|
| Figure 2(一隻貓 0→608 筆) | 第一頁 teaser 圖 |
| Table 0 + Figure 1(12 列版本) | §實驗,改成 ablation 表 |
| §2 五張比較表 | 散文 + 只留 §2.6 那張 |
| §3.2 MAE / soft-argmax 公式 | 附錄 |
| §3.3 AUROC 重現性說明 | 附錄 |
| §4.1 painter、Table 4a | 主文只留 3 條式子,Table 4a 移到附錄 |
| §4.2 tokenisation + Table 4b | 主文保留 |
| §4.3.1 Trial 的故事 + Table 6c | 方法直接寫,Table 6c 移到 §實驗 |
| Figure 8c / 8d | 附錄 |

- [ ] 篇幅:會議主文約 9 頁(6,000–7,000 字),目前光是 §1–§4 就約 16,500 字 → 主要工作是挑選和搬到附錄。
- [ ] 寫作順序:§4 方法 → §3 資料 → §實驗 → §2 → §1 導論 → **摘要最後寫**。

## ℹ️ 投 arXiv 的實務

- [ ] 第一次在 cs 類別投稿通常需要 **endorsement** → 問 DigiPen 的教授。
- [ ] 分類:主類 **cs.CV**,cross-list cs.LG、cs.GR。
- [ ] 資料授權:AFHQ 是 CC BY-NC 4.0;CAT dataset 的授權要查;論文裡要寫明。

### 2026-10-19 — tidy-up step and sketch plan, first runs (6 cats of mac/shard_010, CPU)
- `tools/repaint_tidy.py`: drop fine strokes in the uncertain region, keep strokes per layer (and per region in fur /
  dark hairs / light hairs, order ears → forehead → left cheek → right cheek → chin and mouth), re-lay by contrast,
  refit 100 steps. Blending layers kept whole; topping up short regions not done yet.
  Cat-pixel L1 (6 cats): original 0.0099–0.0303 (1,848 strokes); re-lay + refit, no pruning 0.0096–0.0296.
  Share rule ("hold q of the total"): q 0.93 → 1,315–1,401 strokes, q 0.85 → 1,083–1,166 — about the same count for
  every cat, so it does not express "enough". Absolute rule (keep a stroke contributing ≥ τ): τ 0.5 → 880–1,237,
  τ 1 → 652–916 (white cat 652, calico 916), τ 2 → 519–669. At τ 1 the block-in layers keep all their strokes and the
  hair layers most lose (dark hairs 52–92 of 500, light hairs 2–85): the contribution is area × opacity × colour
  difference, which favours wide strokes; the owner's per-layer criteria (fine-line coverage for hairs, texture for
  fur) are not implemented yet. Figures: `runs/repaint_tidy/compare_zh.png`, `regions_zh.png`.
- `tools/sketch_plan.py`: pose from a 3D mean head recovered by factorisation of 6,000 AFHQ point sets; head outline
  from tri 1/0 (no line at the uncertain chin); eye outlines by SAM on zoomed crops with several prompts (12 of 12
  right by eye; one prompt at the point gave 10 of 12 — the point sat on the fur above the eye on 2 cats); pupils
  (about 9 of 12 plausible), nose by SAM (5 of 6; one took the white bridge), mouth by a smooth dark-line path.
  A: 296–362 plan tokens per cat. B: faint sketch rendered. Pose is NOT reliable yet: roll correlates 0.76 with the
  eye line, yaw −0.81 with the mouth's sideways offset, but cat 000080 (near frontal) gets yaw +41°; refining the eye
  points does not change it, so the mobile ears (6 of the 9 points) drive it. Figure: `runs/sketch_plan/six/figure_zh.png`.
- (later the same day) Owner picked tau 1. Measured the hairs by the painting's own fine lines (recall = sum min(photo
  lines, painting lines) / photo lines) instead of stroke footprints (footprints said 86-97% "covered", far too
  lenient). Full 1,848 strokes: dark 0.44-0.72, light 0.29-0.46 (6 cats). tau 1 loses 0.08-0.15 / 0.10-0.23; a
  footprint-gain rule for hairs did not help. Hair layers x2 / x4 on 3 tabbies (CPU, `tools/hair_count_test.py`):
  x4 dark 0.63-0.74, light 0.49-0.68, L1 0.051-0.060 -> 0.035-0.038, so the stroke count, not the canvas, is the
  limit. Simple cats at tau 1: 562-599 strokes. All of it is zh notebook section 17 (192 cells, runs clean).
- (evening) L5 then L8 (owner: L8 must be able to start fully from L5's weights). Fixed before L5 trains:
  one vocabulary for both (14,561; L8's LAYER5-7 and REGION_END appended after the fields, so the 318 L5 sequences
  already prepared keep their ids and pass the final grammar); one grammar with n_layers 5 or 8 (detail layers region
  by region, REGION_END after each of 5 regions); RoPE only, no learned absolute positions (max length is a buffer,
  not a parameter); same width and depth. L8 sequences (no refit yet): 9,531-9,655 tokens, round trip exact; the L5
  grammar rejects them after layer 4. Still to do for L8: refit + bin the three hair layers on the binned five
  (binning hairs without refit: L1 0.0240 -> 0.0304, 8 cats); training code for variable length and grammar masks.
- (2026-10-20) L5 trained on Kaggle (one T4, `brush-gpt-l5-train`): 29,941,760 parameters, 4,876 train / 291 val cats,
  16 sequences a step. Best val 3.494 at step 2,100 (train batch 3.328), then overfitting; stopped by patience at
  2,900 after 79 min (each train cat seen about 7 times by the best step; no mirroring yet). Per-field val at best:
  paint 5.37, x 4.29, y 3.63, angle 3.04, size 3.06, plan 1.18, end 0.08. Samples: 62 of 64 end within 3,072 tokens
  on Kaggle, 48 of 48 at temperatures 0.8 / 0.6 on the Mac. Heads, ears and coat patterns right; confetti of small
  blocks at temperature 1; eyes rarely drawn. Figure `runs/brushgpt/L5/samples_by_temperature_zh.png`.
- Eye judge, same as J and K (n 128, seed 3, temperatures 1.0 / 0.8 / 0.6; `tools/eye_judge_l.py`): L5 0.445 / 0.460 /
  0.455 vs J 0.500 / 0.514 / 0.487; ratio to own training paintings L5 0.894 / 0.926 / 0.915 vs J 0.932 / 0.956 /
  0.906. References on L's first 128 val cats: photos 0.510, L5 paintings 0.497, L8 paintings 0.508, so the gap is
  the model, not the five layers. L5's eyes are NOT better than J's. Figure `runs/brushgpt/L5/eyes_J_vs_L5_zh.png`
  (`tools/eye_compare_jl.py`). Untested: whether generated strokes follow the model's own eye outlines. FID not run.
- L8 sequences done (`tools/prepare_l8.py`, Mac MPS, 2.4 s a cat, a fresh process every 50 cats because one process
  slows from 2 to 5 s by its 200th cat): hair strokes centred in tri 2 dropped, all others kept (392 / 492 / 483 of
  400 / 500 / 500 on average), refit 50 steps on the binned five, binned, written by region. Each starts with its L5
  sequence token for token (asserted). 8,426-9,721 tokens (max 9,728: only 7 to spare). L1 to the photo (cat area)
  median 0.058 -> 0.034, lower on all 5,547. `data/processed/brush_l/data_L8.npz` built (rebuilt there on 10-20 after tools/kaggle_brush_l.py cleared data/kaggle/brush_l8/data/). Not trained yet;
  owner to choose: mirroring first or L8 now. All of it is zh notebook sections 17.13-17.16, Table 0, figure 1.
