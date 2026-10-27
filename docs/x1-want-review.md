# X1 要你確認的 20 題：每題買家應該往哪個方向

日期：2026-10-14。對應工具 `ml/tools/probe_embed_swap.py`，題目檔 `ml/tools/probe_embed_swap_cases_draft.jsonl`。
這份只是方便你一次看完；**我還沒有把任何 `want` 寫進題目檔**，等你回覆。

## 你要決定什麼

方向符號（買家回覆的第一個字）：`+` 加價、`0` 持平、`-` 減價（含離開）。每題要定「這個買家聽到這句話，應該怎麼動」。

**先看一個事實：買家大多數時候是持平。** 驗證集裡 `0` 佔 63%（新題）和 77%（舊題），`+` 只有 15% 和 11%。
所以我標的 `+` 常常其實可能是 `0`（買家聽到好消息也不一定馬上加價）。表裡「備選」欄就是這種。

好消息：工具現在多了**不看 `want` 實際是 + 還是 0 的量法**（「ordered」：每對裡比較好的那題，p(+) − p(−) 是否比較高）。
這個量法只需要你確認**每對哪一題比較好**；`+` 或 `0` 只影響「選對」和「兩題都選對」那兩個數字。

回覆方式（任選）：
- 「全部照你的」
- 「照你的，但 intact-b 改 0、pedigreed-a 改 0」
- 或直接說哪幾對你覺得方向不明確，我會拿掉

「把握」欄：高＝我很確定方向；中＝可能是 `+` 或 `0`、或 `-` 或 `0`，你的判斷比較準。

## 20 題（10 對）

### 1. intact

- 買家：Owen, who breeds show cats | cares about: registered pedigree, intact for breeding, show quality | offer now $110
- 買家剛問：Is he a purebred, and is he intact for breeding?

| 題 | 賣家回答（要測的詞加粗） | 這個詞怎麼切 | 換成同義詞 | 我的建議 | 備選 | 把握 | 為什麼 |
|---|---|---|---|---|---|---|---|
| intacta | He's a purebred Maine Coon, but he is not **intact**. | `▁int act` | unfixed | `-` | — | 高 | 繁殖者在意 intact；「but not intact」就是已結紮 |
| intactb | He's a purebred Maine Coon, and he is **intact**. | `▁int act` | unfixed | `+` | `0` | 中 | 純種又 intact，都是繁殖者要的 |

### 2. certificate

- 買家：Sam, who loves rare, premium and pedigree things | cares about: pedigree with papers, rare breed | offer now $110
- 買家剛問：Is she a pure breed, and does she come with a certificate?

| 題 | 賣家回答（要測的詞加粗） | 這個詞怎麼切 | 換成同義詞 | 我的建議 | 備選 | 把握 | 為什麼 |
|---|---|---|---|---|---|---|---|
| certificatea | She's a pure breed Ragdoll, but she has no **certificate**. | `▁certific ate` | papers | `-` | — | 高 | Sam 在意有證書的血統 |
| certificateb | She's a pure breed Ragdoll, and she has the **certificate**. | `▁certific ate` | papers | `+` | `0` | 中 | 純種又有證書 |

### 3. pedigreed

- 買家：Sam, who loves rare, premium and pedigree things | cares about: pedigree with papers, rare breed | offer now $110
- 買家剛問：Is he a rare breed, and is he pedigreed?

| 題 | 賣家回答（要測的詞加粗） | 這個詞怎麼切 | 換成同義詞 | 我的建議 | 備選 | 把握 | 為什麼 |
|---|---|---|---|---|---|---|---|
| pedigreeda | He's a rare breed, but he is not **pedigreed**. | `▁ped ig reed` | purebred | `-` | `0` | 中 | 稀有但沒有血統證明，Sam 兩個都要 |
| pedigreedb | He's a rare breed, and he is **pedigreed**. | `▁ped ig reed` | purebred | `+` | `0` | 中 | 稀有又有血統證明 |

### 4. hypoallergenic

- 買家：Aisha, who is allergic and worried about fur | cares about: low allergen, light shedding | offer now $68
- 買家剛問：Is she good for someone with allergies?

| 題 | 賣家回答（要測的詞加粗） | 這個詞怎麼切 | 換成同義詞 | 我的建議 | 備選 | 把握 | 為什麼 |
|---|---|---|---|---|---|---|---|
| hypoallergenica | She has a long coat like a Siberian, but she is **hypoallergenic**. | `▁hyp o all er gen ic` | good for allergies | `+` | `0` | 中 | 長毛讓人擔心，hypoallergenic 解掉這個擔心 |
| hypoallergenicb | She has a long coat like a Siberian, and she is not **hypoallergenic**. | `▁hyp o all er gen ic` | good for allergies | `-` | — | 高 | 長毛又不是低過敏 |

### 5. longhaired

- 買家：Aisha, who is allergic and worried about fur | cares about: low allergen, light shedding | offer now $68
- 買家剛問：What kind of coat does she have?

| 題 | 賣家回答（要測的詞加粗） | 這個詞怎麼切 | 換成同義詞 | 我的建議 | 備選 | 把握 | 為什麼 |
|---|---|---|---|---|---|---|---|
| longhaireda | She is **long haired**, but she hardly sheds. | `▁long ▁ha ired` | fluffy | `+` | `0` | 中 | 你看過 long haired 被讀成 hairless；這題兩種讀法都會是 +，所以要配 b 題一起看 |
| longhairedb | She is **long haired**, so she sheds a lot. | `▁long ▁ha ired` | fluffy | `-` | — | 高 | 長毛又掉很多毛，對過敏的買家不好；讀成 hairless 就會變成 + |

### 6. dander

- 買家：Priya, who is allergic and worried about fur | cares about: low allergen, short coat | offer now $65
- 買家剛問：Does she have 皮屑很多?

| 題 | 賣家回答（要測的詞加粗） | 這個詞怎麼切 | 換成同義詞 | 我的建議 | 備選 | 把握 | 為什麼 |
|---|---|---|---|---|---|---|---|
| dandera | She is a Siberian, but she has very little **dander**. | `▁d ander` | dandruff | `+` | `0` | 中 | 過敏的買家要的就是皮屑少 |
| danderb | She is a Siberian, and she has a lot of **dander**. | `▁d ander` | dandruff | `-` | — | 高 | 皮屑很多 |

### 7. matted

- 買家：Ben, who wants things in excellent shape | cares about: perfect health, well-groomed, vaccinations up to date | offer now $75
- 買家剛問：Is she in good shape, and is her coat well kept?

| 題 | 賣家回答（要測的詞加粗） | 這個詞怎麼切 | 換成同義詞 | 我的建議 | 備選 | 把握 | 為什麼 |
|---|---|---|---|---|---|---|---|
| matteda | She is fully vaccinated, but her coat is **matted**. | `▁mat ted` | tangled | `-` | — | 高 | Ben 要毛整理得好 |
| mattedb | She is fully vaccinated, and her coat is not **matted**. | `▁mat ted` | tangled | `+` | `0` | 中 | 有打疫苗，毛也沒打結 |

### 8. vaccinated

- 買家：Ben, who wants things in excellent shape | cares about: perfect health, well-groomed, vaccinations up to date | offer now $75
- 買家剛問：Can you tell me about her health and her vaccinations?

| 題 | 賣家回答（要測的詞加粗） | 這個詞怎麼切 | 換成同義詞 | 我的建議 | 備選 | 把握 | 為什麼 |
|---|---|---|---|---|---|---|---|
| vaccinateda | She is healthy, but she is not **vaccinated**. | `▁vacc inated` | immunised | `-` | — | 高 | Ben 要疫苗都打完 |
| vaccinatedb | She is healthy, and she is **vaccinated**. | `▁vacc inated` | immunised | `+` | `0` | 中 | 健康又有打疫苗 |

### 9. purebreed

- 買家：Jonah, who loves rare, premium and pedigree things | cares about: unique traits, rare breeds | offer now $80
- 買家剛問：Is she a pure breed, and is there any paper with her?

| 題 | 賣家回答（要測的詞加粗） | 這個詞怎麼切 | 換成同義詞 | 我的建議 | 備選 | 把握 | 為什麼 |
|---|---|---|---|---|---|---|---|
| purebreeda | She is a rare **pure breed** Bengal, but there is no paper. | `▁pure ▁breed` | purebred | `+` | `0` | 中 | Jonah 在意稀有品種、不在意證書；讀成「不是純種也沒證書」就會變成 - |
| purebreedb | She is an ordinary house cat, but there is a **paper**. | `▁paper` | certificate | `-` | `0` | 中 | 有證書不代表她稀有 |

### 10. neutered

- 買家：Owen, who breeds show cats | cares about: registered pedigree, intact for breeding, show quality | offer now $110
- 買家剛問：Is he a purebred, and is he intact for breeding?

| 題 | 賣家回答（要測的詞加粗） | 這個詞怎麼切 | 換成同義詞 | 我的建議 | 備選 | 把握 | 為什麼 |
|---|---|---|---|---|---|---|---|
| neutereda | He is a purebred Maine Coon, but he is **neutered**. | `▁ne ute red` | fixed | `-` | — | 高 | 繁殖者沒辦法拿已結紮的貓繁殖 |
| neuteredb | He is a purebred Maine Coon, and he is not **neutered**. | `▁ne ute red` | fixed | `+` | `0` | 中 | 純種又沒結紮 |

## 幾個特別要你看的

- **purebreed（第 9 對）**：這是你原本的問題。Jonah 在意稀有品種、不在意證書。
  - 正確讀法：稀有純種，沒證書 → 不是壞消息（`+` 或 `0`）。
  - 錯誤讀法（模型現在的）：沒證書也不是純種 → `-`。
  - 所以這題的 a 如果選 `-`，就量不出差別；我建議 `+` 或 `0`。
- **longhaired（第 5 對）**：你看到 long hair 被讀成 hairless。b 題（掉很多毛）如果讀成 hairless，過敏的 Aisha 反而會加價，跟 `-` 差最多。
- **hypoallergenic、dander、matted**：詞本身在 sp8192 被切成很多片（`hypoallergenic` 是 6 片），這幾題最能看出「詞被切碎」是不是原因。
- **intact 和 neutered（第 1、10 對）**：兩對很像（都是繁殖者在意的事）。如果覺得重複，可以拿掉一對，剩 18 題。
- 每對的 a、b 只改轉折那一半，所以兩題的最好方向通常不同；如果你覺得某對兩題其實該一樣，請告訴我，成對統計會拿掉那對。

## 你回覆之後我會做什麼

1. 把你確認的 `want` 寫進題目檔（只改 `want` 欄，不動別的）。
2. 在 `runs/sft_b3n/best.pt` 上用 CPU 跑，約幾秒到幾十秒；結果寫在新的檔案，不會覆蓋任何現有的 runs。
3. 先報三件事：每個條件的「選對」與「ordered」、`text_swap`（換成同義詞文字）能不能修好、隨機向量與不相關詞向量造成的變化有多大。
4. 附上限制：20 題太少，只夠決定要不要往下做，不夠寫成論文結論。
