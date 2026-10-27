# B4f9 計畫:場景改成「先寫句子、再列清單」

日期:2026-10-16。**程式已改好(diff:`ml/runs/sft_b4f9/code_changes.diff`),資料已轉好,還沒訓練。** 刷筆 GPT J 跑完、你說開跑才開始。
這份裡的數字都是在這台機器上實際跑出來的,出處寫在每個數字旁邊。

## 一句話

B4 的場景(imagine)大多不合格,原因是**句子沒講到自己清單裡的東西**。
這次只改一件事:把場景的寫法從「清單 | 句子」改成「句子 | 清單」。
其餘全部照 B4:同一批買賣、同一個訓練/驗證切分、同一套訓練配方。這樣量出來的差別就只能是這個改動造成的。
新格式叫 **prompt format 9**,模型叫 **B4f9**(命名照 B2f6 的慣例:B2 的資料換成 format 6)。

## 為什麼要改

### 1. 主要的錯是「沒講到」

`tools/probe_logit_bias.py`(2026-10-16,在 CPU 上跑,溫度 0.7、top-k 50,新集 152 題)把每個場景**所有**違反的規則都算進去
(evaluation 用的 `imagine_problem` 只回報第一條):

| | 不合格 | 句子沒講到清單裡的某樣東西 | 只錯這一條 |
|---|---|---|---|
| B3(遊戲現在用的) | 118 / 152 | 110 | 76 |
| B4 | 101 / 152 | 95 | 64 |

### 2. 取樣時加限制修不好

同一支工具在取樣時禁掉違規的 token(`logit_bias`)。結果:清單錯字、多提沒選的東西、提到畫不出來的東西,
三類幾乎都降到 0,**但合格率沒有超出換種子造成的波動**(B4 舊集換一個種子就差 17 個百分點:49.4% 對 66.3%)。
禁字只能「不准說」,沒辦法「叫它說」;改成「沒講完不准結束」,模型會一直寫下去、自己生出下一回合
(新集 B3 有 66 題、B4 有 82 題這樣)。
而且依 [`CLAUDE.md`](../CLAUDE.md),取樣時的禁字清單本來就是「禁止」那一類。**能修的地方只剩資料和訓練。**

### 3. 為什麼「句子先寫」可能有用(假設,還沒驗證)

現在的格式要模型**先規劃**三樣東西,再寫一句話**照著規劃講完**。7,100 萬參數的模型寫到句子中段,
常常就照習慣寫下去,不再回頭看清單(notebook 6 讀過 B1 也是照習慣寫場景)。
反過來寫的話,清單是從**自己剛寫好、就在眼前的句子**抄出來的。抄近處的字,通常比照遠處的計畫寫完整件事容易。

⚠️ 這只是假設,可能錯。可能的壞結果寫在「風險」那一節。

## 格式 9 長什麼樣

B4 訓練資料裡真實的一題(seed 2000,Hana):

```
format 5(B4): skyscraper, stairs, pillow | Up four flights of stairs in my little studio, she could watch the skyscraper lights at night and sleep on a pillow by my feet.<|end|>
format 9(B4f9): Up four flights of stairs in my little studio, she could watch the skyscraper lights at night and sleep on a pillow by my feet. | stairs, skyscraper, pillow<|end|>
```

prompt 一個字都不改(一樣以 `<|imagine|>` 結尾),只改目標。

### 兩個附帶的改動(都是格式本身需要的,不是另外加的實驗)

**① 清單照句子裡出現的先後排。** B4 訓練資料的 3,740 個場景裡,只有 **2,070 個(55%)** 的清單順序
剛好等於句子裡出現的順序(上面 Hana 那題就不是)。句子先寫的話,清單照出現順序排,模型才是一路從前往後抄;
不照順序的話,它要在句子裡跳來跳去找。
「出現在哪裡」用規則自己的判斷(`stories._named`:看東西名字的最後一個字,例如 `alarm clock` 看 `clock`)。

**② 這個格式的生成上限從 64 個 token 提高到 96。** B4 訓練目標的長度:中位數 50、最長 85,
訓練集有 **126 / 3,740(3.4%)** 超過 64、驗證集 17 / 466。
舊格式超過上限時被截掉的是句子尾巴;新格式被截掉的會是**清單**,整個場景就讀不到了。
96 夠放最長的 85,而且最長的 prompt 603 + 96 = 699,還在訓練用的 768 以內。
舊格式(5–8)維持 64,舊模型的 evaluation 才能跟以前逐位元相同。

## 資料:直接從 B4 的檔案轉,不重新組

今天用 `sales_sft --sources b4` 重組,會得到 **49,809** 個訓練例子,B4 當時是 **48,832** 個:
`sales_b2.jsonl` 之後又多寫了買賣(B4 是 10-13 組的)。
所以 B4f9 的資料**不重組**,而是讀 B4 的 `data/processed/buyers_b4/{train,val}.jsonl`,只改 imagine 的目標:

- 拆成清單和句子,清單依出現順序重排,寫成 `句子 | 清單<|end|>`;
- 其他例子(open / ask / react)原封不動照抄;
- 轉完要檢查三件事:非 imagine 的例子跟 B4 **逐行相同**;imagine 的 prompt 逐字相同;
  轉出來的每個場景都過得了 `imagine_problem`(B4 的 3,740 + 466 個本來都過)。

B4 檔案的雜湊(前 16 碼):train `b738e4484b798c2f`、val `b7ee1c0a3ff5879e`。

## 要改的程式

| 檔案 | 改什麼 |
|---|---|
| `buyers/format.py` | 加 `LINE_FIRST_FORMATS = (9,)` 與說明 |
| `buyers/stories.py` | 加 `mention_order(things, says)`:用 `_named` 同一套判斷,傳回依出現順序排好的清單;加 `imagine_kinds(row)`:一個場景違反的**每一條**規則 |
| `buyers/sales_sft.py` | `imagine()` 在 format 9 寫成 `句子 | 清單`;`--prompt-format` 接受 9(之後新的買賣也能直接組) |
| `buyers/cat_market.py` | `parse()` 多收一個 `prompt_format` 參數:format 9 時清單在**最後一個** ` \| ` 之後。`MAX_NEW` 依格式決定(format 9 的 imagine 是 96)。遊戲的 `_say` 把 `self.gpt.prompt_format` 傳進去 |
| `tools/eval_buyer_b1.py` | 傳 `model.prompt_format` 給 `parse` 和 `MAX_NEW`;imagine 的報告多一欄「每一類錯各幾題」(`stories.imagine_kinds`);報告記下 `seed_offset` |
| `tools/make_line_first.py`(新) | 上面「資料」那一節的轉換與三項檢查 |
| `tools/compare_b4f9.py`(新) | 讀 B4 / B4f9 各 3 個種子的報告,印對照表,照下面的判準給結論 |
| `tools/run_b4f9_mac.sh`(新) | J 還在跑就拒絕開始 → 轉資料 → 訓練 → 量測(下面)。**手動開跑**(你決定) |
| tests | `test_sales_sft.py`:format 9 的目標長相與順序;`test_cat_market.py`:format 9 的 `parse` 讀得回清單,format 5 的結果不變 |

⚠️ **清單一定要由模型自己寫出來。** 遊戲不能拿句子去比對字典、把東西找出來當清單,
那就是 CLAUDE.md 禁止的關鍵字比對。

## 訓練

照 B4 的配方,只換資料夾:

```
cd ml && PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7 PYTHONPATH=. \
python -m sellanything_ml.llm.sft --init runs/pretrain_large/final.pt --data data/processed/buyers_b4f9 \
    --run-dir runs/sft_b4f9 --epochs 3 --batch 16 --lr 3e-4 --eval-every 500
```

- **要多久**:B4 是 9,156 步、每步約 1.6 秒,大約 4 小時。B4f9 例子數一樣,應該差不多。
- **什麼時候跑**:一次只跑一個 GPU 工作。刷筆 GPT J 的 driver(pid 27235)會訓練、再評估,
  上限 2,400 分鐘,每 100 步約 25 分鐘。`run_b4f9_mac.sh` 會先等這個 driver 結束才動。
  **要不要等它結束就自動開跑,還是等你說一聲,由你決定**(見最後一節)。
- **驗證損失不能跟 B4 比**:imagine 目標的順序變了,算的不是同一件事。

## 怎麼判斷有沒有用(先寫好,跑完不改)

上次量到換一個種子就差 17 個百分點,所以**每個數字都跑 3 個種子**(`--seed-offset 0 / 1 / 2`),
B4 和 B4f9 都跑,用同一份 prompt(`runs/eval_sets_b2/val_{old,new}_format5.jsonl`;
imagine 的檢查不看目標,所以 format 9 也能用同一份)。

**主要指標**:新集 imagine 合格率,3 個種子的平均。

- **算有用**:B4f9 的平均比 B4 高 10 個百分點以上,**而且** B4f9 最低的那個種子,
  比 B4 最高的那個種子還高。
- **算沒用**:不符合上面的條件,就照實寫「沒用」。不會換指標、不會只挑好看的種子報。

**要一起看的**:

1. 各類錯的題數(沒講到 / 清單錯字 / 多提沒選的 / 提到畫不出來的 / 數量不對),看錯誤是消失了,還是**換到別類**;
2. open / ask / react 的成績不能明顯變差,因為它們共用同一個模型(比對 evaluation 報告的各欄);
3. 新集隨機讀 24 個場景,寫下好幾個、不好幾個、壞在哪裡,跟 devlog 10-11 的讀法一樣。
   合格不等於好:句子可能變得很怪,只是剛好提到三樣東西。

量完的結果寫進 `docs/model-versions.md`(B4f9 一節)和當天的 devlog。

## 風險:可能會怎麼失敗

- **錯誤換地方**:句子先寫,模型可能在句子裡寫到畫不出來的東西(sofa、window),然後清單照抄,
  變成「清單錯字」或「提到畫不出來的」。所以才要看第 1 項的分類,而不只看合格率。
- **少了規劃,句子變散**:舊格式先選好三樣東西,句子至少有個方向;新格式可能寫出只提到兩樣、
  或提到四樣的句子。訓練資料的句子都剛好三樣,但模型學不學得到不確定。
- **沒有效果**:如果錯是出在「不知道哪些東西畫得出來」,而不是出在順序,換順序就沒用。
  那就表示該補的是資料(例如更多冷門東西的場景;B2f6 量過:訓練裡只出現 1–9 次的東西,模型幾乎不會用)。

## 你要決定的事

1. **名字和格式號**:B4f9 / format 9,可以嗎?
2. **兩個附帶改動一起測**(依出現順序排清單、上限 96),還是要拆開?
   我建議一起測:上限不提高,約 3% 的場景會在新格式下整個讀不到;順序是新格式的一部分。
   拆開要多訓練一次,大約多 4 小時。
3. **什麼時候開跑**:J 的 driver 一結束就自動開始,還是等你說?
4. 程式改動寫好之後,要不要先給你看 diff,再開始訓練?
