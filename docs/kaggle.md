# Kaggle：怎麼送工作上去、怎麼把結果下載回來

> 2026-10-19 重畫 AFHQ 時踩過的坑都在這裡，照第 4 節做就不會再卡。
> 所有指令都在 `sell-anything/ml/` 底下執行。

---

## 1. 帳號與工具

| 項目 | 值 |
|---|---|
| Kaggle 帳號 | `alinachuang` |
| CLI | `/opt/anaconda3/bin/kaggle`，版本 **2.2.4**（`kaggle --version`） |
| API token | `~/.kaggle/access_token`（CLI 2.x 讀這個檔或環境變數 `KAGGLE_API_TOKEN`）。**沒有** `~/.kaggle/kaggle.json` |
| GPU 配額 | 每週 **30 小時**，週六 00:00 UTC 重算。網頁 https://www.kaggle.com/settings 的 **Quotas → Kaggle GPU** 看已用多少 |
| 可以用的 GPU | 免費的 Kaggle / Colab 可以；**不准付費**（`docs/plan.md` Ground rules） |

⚠️ **金鑰與密碼一律由擁有者自己處理**，不寫進任何腳本或文件。
要看私人 kernel 的網頁時，在內建瀏覽器打開登入頁，由擁有者自己登入。

---

## 2. 這次重畫用到的 Kaggle 東西

| 名稱 | 種類 | 內容 |
|---|---|---|
| `alinachuang/brush-repaint-afhq` | 資料集（570 MB） | AFHQ 重畫輸入 shard 0–9（`repaint_prep.py` 產生）＋舊版程式碼 |
| `alinachuang/brush-repaint-afhq-todo` | 資料集（355 KB） | 有 `--only` 的新版程式碼＋ `todo_gpu0.txt`、`todo_gpu1.txt`（還沒畫的貓） |
| `alinachuang/brush-repaint-cat` | 資料集 | CAT 重畫輸入 shard 11–30（Colab 改用第 8 節的 Drive 套件） |
| `alinachuang/brush-repaint-afhq-t4` | kernel（T4 ×2） | **第一輪**：680 分鐘、畫了 **3,984 隻**（shard 0–3、5–8）。本機副本 `data/processed/repaint/kaggle_t4_run1/out/` |
| `alinachuang/brush-repaint-afhq-t4-b` | kernel（T4 ×2） | **第二輪**：只畫 `--only` 清單上的貓（2026-10-19 13:20 左右開始） |
| `alinachuang/brush-repaint-afhq-pack` | kernel（只用 CPU） | 把第一輪的輸出每個 shard 打成一個 tar（第 4 節的做法） |

本機的 kernel 與資料集設定檔都在 `ml/data/kaggle/`（`.git/info/exclude` 已排除，不會進 git）：
`repaint_afhq/`、`repaint_afhq_todo/`、`repaint_pack/`、`repaint_cat/`、`digital_timing/`。

---

## 3. 踩過的坑（一定要讀）

| # | 症狀 | 原因 | 做法 |
|---|---|---|---|
| 1 | `kaggle kernels output` 下載到一半就**安靜地停了**（2,264 / 3,984 個檔），沒有錯誤訊息 | 輸出檔案的清單是**一頁 200 個、分頁請求**，請求太多被限制 | 不要直接下載幾千個檔案，改用第 4 節的打包法 |
| 2 | `429 Client Error: Too Many Requests … ListKernelSessionOutput` | 同上；而且**一直重試會讓限制持續**（重試了 30 分鐘都是 429） | 停掉重試，等 1 小時以上；或改用打包法（只要列一頁） |
| 3 | 用 CLI 列出來的檔案數比實際少（列出 3,773，實際 3,984） | 分頁的最後幾頁沒有回來 | **不要用清單數字當「畫了幾隻」**，以打包 kernel 的 `summary.json` 或解壓後實際檔案為準 |
| 4 | 重跑 `kaggle kernels output` 會**從頭重新下載**，不會跳過已經有的檔案 | CLI 沒有續傳 | 用 `--file-pattern` 只下載缺的部分 |
| 5 | 網頁 Output 分頁只顯示約 **500 個檔案**；「Output actions → New dataset」建出來的資料集也只含這些 | 網頁的顯示上限 | **不要用網頁建資料集來下載大量輸出** |
| 6 | 新 kernel 一直 **QUEUED**（排隊超過 1 小時） | Kaggle 的 T4 ×2 機器剛好很搶 | 先在網頁左下角 **View Active Events** 確認沒有別的東西在跑、在 Settings 看配額；兩者都正常就是 Kaggle 那邊排隊。改讓 Mac 從清單的另一端開始畫，不要乾等 |
| 7 | 推新版本到同一個 kernel 名稱 | **舊版本的輸出會被新版本取代** | 舊輸出還沒完整下載前，新工作一律用**新的 kernel 名稱**（例如 `-b`） |
| 8 | 資料集解壓後少了最上層的 `ml/` 資料夾 | Kaggle 解壓的方式 | kernel 裡用 `glob("/kaggle/input/**/tools/repaint.py", recursive=True)` 找程式碼，不要寫死路徑 |
| 9 | 下載下來的 `.npz` 有一個壞掉（打不開） | 中途停掉的下載留下不完整的檔案 | 合併前一定跑 `repaint_merge.py --check`（逐一打開檢查），壞的從 tar 重新取出 |

---

## 4. ⭐ 下載大量輸出的標準做法：先在 Kaggle 上打包

原理：開一個**只用 CPU、不耗 GPU 配額**的小 kernel，把要下載的那個 kernel 的輸出接進來當輸入，
每個 shard 打成一個 tar。輸出只剩十幾個檔案，列清單只要一頁，不會被限制。

**步驟 1：改設定檔，指定要打包哪個 kernel 的輸出**

`ml/data/kaggle/repaint_pack/kernel/kernel-metadata.json` 的 `kernel_sources` 換成目標 kernel，
例如要打包第二輪：

```json
"kernel_sources": ["alinachuang/brush-repaint-afhq-t4-b"]
```

（`run.py` 會找輸入裡所有 `out/shard_*` 資料夾，不用改。若要保留舊的打包結果，`id` 也換一個新名稱，例如 `brush-repaint-afhq-pack-b`，因為第 3 節第 7 點。）

**步驟 2：送出並等它跑完**（CPU，約幾分鐘）

```bash
kaggle kernels push -p data/kaggle/repaint_pack/kernel
```

```bash
until kaggle kernels status alinachuang/brush-repaint-afhq-pack | grep -q "COMPLETE\|ERROR"; do sleep 120; done
```

**步驟 3：只下載需要的 tar**（`--file-pattern` 是正規表示式）

```bash
kaggle kernels output alinachuang/brush-repaint-afhq-pack -p data/processed/repaint/kaggle_t4_run1/packs --file-pattern "summary.json|shard_00[3678]\.tar$"
```

`summary.json` 寫著每個 shard 實際有幾隻，這才是真正的數字（第 3 節第 3 點）。

**步驟 4：解壓縮到輸出資料夾**（`-k`：已經存在的檔案不覆蓋）

```bash
tar -xkf data/processed/repaint/kaggle_t4_run1/packs/packs/shard_006.tar -C data/processed/repaint/kaggle_t4_run1/out
```

**步驟 5：檢查與合併**（第 5 節）

---

## 5. 合併、檢查、分配還沒畫的貓

```bash
PYTHONPATH=.:tools python tools/repaint_merge.py --done data/processed/repaint/mac data/processed/repaint/kaggle_t4_run1/out --check
```

- `--done` 後面列出所有輸出資料夾（Mac、Kaggle 各輪）；同一隻貓有多份時，**排在前面的資料夾優先**，其他列為重複。
- `--check` 逐一打開每個檔案，八層都在才算完成；壞的會列在 `bad files`。
- 加 `--todo-out <資料夾>` 會寫出還沒畫的清單：`todo_gpu0.txt`、`todo_gpu1.txt`（給 Kaggle 兩張 GPU）與 `todo_mac.txt`。

**分工的規則：Kaggle 往前、Mac 往後，在中間會合。**
- Kaggle：`repaint.py --only todo_gpuN.txt`，依 shard 由小到大畫（`data/kaggle/repaint_afhq_todo/kernel/run.py`）。
- Mac：`tools/run_repaint_mac.sh`，用 `--reverse --reverse-cats --only data/processed/repaint/todo_mac_all.txt`，從最後一隻往前畫，
  每 25 隻重開一次 python（避免 MPS 記憶體越用越多）。**清單檔每批重新讀取，所以直接改 `todo_mac_all.txt` 就能改 Mac 要畫什麼，不用重開。**
- 兩邊重疊畫到的貓，合併時只留一份。

---

## 6. 送新的 GPU 工作上去

1. 程式碼與小檔案做成**新的小資料集**，不要為了改程式碼重新上傳 570 MB 的輸入：

   ```bash
   kaggle datasets create -p data/kaggle/repaint_afhq_todo/data -r zip
   ```

   之後要更新同一個資料集用 `kaggle datasets version -p <資料夾> -m "說明" -r zip`。
   等它 ready：`kaggle datasets status alinachuang/<名稱>`。
2. kernel 設定檔 `dataset_sources` 同時列出輸入資料集與程式碼資料集；GPU 工作設 `"enable_gpu": true`、`"machine_shape": "NvidiaTeslaT4"`（兩張 T4）。
3. kernel 裡每個 GPU 各開一個 process（`CUDA_VISIBLE_DEVICES=0/1`），設 `--stop-after-min` 在 12 小時的上限前乾淨地停下（第一輪用 680 分鐘）。
4. 送出：`kaggle kernels push -p <kernel 資料夾>`；查狀態：`kaggle kernels status alinachuang/<名稱>`（查狀態沒有被限制過）。
5. **少呼叫列檔案的 API**：不要寫每幾分鐘重試一次下載的迴圈（第 3 節第 2 點）。

---

## 7. 2026-10-19 16:30 的狀態：AFHQ 重畫完成

- **AFHQ 5,547 隻全部完成**，`repaint_merge.py --check` 逐一檢查 0 個壞檔：
  第一輪 `data/processed/repaint/kaggle_t4_run1/out/` 3,984 隻、第二輪 `kaggle_t4_run2/out/` 627 隻、Mac `mac/` 1,472 隻，重複 536 隻（合併時只留一份）。
- 第二輪的輸出也是照第 4 節打包下載的（打包 kernel `brush-repaint-afhq-pack-b`，`summary.json`：shard 3 有 10、shard 4 有 400、shard 8 有 217）。
- 合併時三個資料夾的順序：`kaggle_t4_run1/out`、`kaggle_t4_run2/out`、`mac`（同一隻貓以前面的為準）。
- Mac 的重畫迴圈已停止。

---

## 8. Colab：重畫 CAT 資料集（2026-10-20 開始）

- 套件：`PYTHONPATH=. python tools/prepare_colab_repaint.py` → `ml/data/colab_repaint/`（`.git/info/exclude` 已排除）：
  `colab_repaint_pkg.zip`（1.04 GB：與 Kaggle 相同的重畫程式＋輸入 shard 11–30）與 `repaint_cat_colab.ipynb`。
- 共 9,714 隻 CAT 貓（shard 11 的 47 隻 AFHQ 已畫好，`--source cat` 跳過）。
- 擁有者把兩個檔放到 Google Drive 的 `我的雲端硬碟/sell_anything_colab/repaint_cat/`，在 Colab 選 T4 GPU、由上往下執行。
  畫作直接寫進 Drive 的 `repaint_cat/out/shard_NNN/`（先寫暫存檔再改名），斷線後重跑會跳過已畫好的貓。
- Colab 從 shard 11 往後畫；若要讓 Kaggle 或 Mac 一起畫，用 `--reverse --reverse-cats` 從 shard 30 往前，兩邊在中間會合。
- 速度：Kaggle 的 T4 每張約 2.9 隻／分鐘；Mac（M5，MPS）每隻約 31 秒。Mac 畫的時候 GPU 被占滿，會影響同時玩遊戲。
- 下載 Drive 上的 `out/` 後，用 `repaint_merge.py --done … --check` 檢查並合併（第 5 節）。
