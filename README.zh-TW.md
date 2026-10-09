# LoRA Tag Studio

[English](README.md) | **繁體中文**

自架的 LoRA 訓練資料集工具：上傳圖片或整個資料夾，自動產生標註（WD14 的 Danbooru 標籤，加上視覺語言模型寫的自然語言描述），在網頁上檢查、修改後，匯出 Civitai、kohya_ss 或 Hugging Face 可以直接訓練的資料集，也可以直接送到 Civitai 雲端訓練。

- **依底模給出正確的標註方式**：SD1.5（動漫 / 寫實）、SDXL、Pony V6、Illustrious、NoobAI、Animagine、Anima、FLUX.1 各有不同的 caption 格式、標籤順序、分級標籤、品質標籤與訓練建議。
- **兩種標註器**：WD14（SmilingWolf v3，ONNX，CPU / GPU）產生 Danbooru 標籤；VLM（Ollama / JoyCaption / Claude / 任何 OpenAI 相容端點）產生自然語言描述，也可以用 JoyCaption 的 Danbooru 模式補充或取代標籤。
- **支援 NSFW**：WD14 完整輸出 NSFW 標籤、各底模分級標籤自動換算、VLM 露骨描述模式、可選的無審查 JoyCaption 服務。
- **WebUI**：拖放上傳資料夾 / zip、圖庫篩選、標籤編輯（拖曳排序、自動完成）、標籤統計、批次加入 / 移除 / 取代、角色特徵修剪、黑名單。
- **waifu2x 放大與降噪**：短邊太小的圖先放大再訓練，LoRA 才不會學到模糊；也可以只去掉 JPEG 雜訊、不改尺寸。原圖會保留，可以還原。
- **角色篩選**：給幾張參考圖，用 CCIP 從一堆圖片挑出某個角色，挑好的圖可以下載或匯入專案。
- **直接送到 Civitai 雲端訓練**：上傳圖片與 caption、試算 Buzz、確認後開始訓練、查看進度，並下載每個 epoch 的 LoRA（Civitai Orchestration API）。
- **給其他 LLM 使用**：REST API（OpenAPI 規格可直接當 tool server）＋ MCP 伺服器（Claude Desktop / Claude Code / Cursor / Open WebUI）。
- **多國語言**：WebUI、錯誤訊息、標註指南與匯出的 README 支援繁體中文、English、日本語、한국어、简体中文，依瀏覽器語言自動切換，右上角可手動選擇。

---

## 需求

- Docker 與 Docker Compose v2。
- NVIDIA GPU（選用）：[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) 與驅動 ≥ 580（onnxruntime-gpu 1.30 為 CUDA 13 版本，支援 RTX 50 系列）。
- 模型第一次使用時才下載，需要的空間：WD14 eva02-large 約 1.2GB、Ollama `qwen2.5vl:7b` 約 6GB、JoyCaption 約 17GB。
- JoyCaption 約需 17GB VRAM（bf16）。

## 快速開始

```bash
git clone https://github.com/sss22213/LoRA-Tag-Studio.git
cd LoRA-Tag-Studio
cp .env.example .env          # 視需要修改，各變數說明見「設定」
docker compose up -d --build  # CPU 版
```

開啟 <http://localhost:7870>。第一次標註時會自動下載 WD14 模型，存在 `models` volume。

| 組合 | 指令 |
|---|---|
| CPU，只用 WD14 標籤 | `docker compose up -d --build` |
| CPU + Ollama 自然語言描述 | `docker compose --profile vlm up -d --build` |
| NVIDIA GPU + Ollama | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile vlm up -d --build` |
| NVIDIA GPU + JoyCaption（NSFW） | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile joycaption up -d --build`（並在 `.env` 改 `VLM_*` 三行，見 [NSFW](#nsfw)） |

> 也可以在 `.env` 設定 `COMPOSE_FILE` / `COMPOSE_PROFILES`，之後只要 `docker compose up -d`。

`--profile vlm` 會啟動 Ollama 並自動下載 `VLM_MODEL`（預設 `qwen2.5vl:7b`）。想換模型：

```bash
docker compose exec ollama ollama pull qwen2.5vl:32b   # 並把 .env 的 VLM_MODEL 改成同名
```

### 停止

```bash
docker compose --profile vlm --profile joycaption down
```

- 就算只啟動了其中一個，也把兩個 profile 都帶上；帶到沒在跑的 profile 不會有影響。不帶的話，`down` 會漏掉 Ollama / JoyCaption 的容器，最後出現 `Network lora-tag-studio_default  Resource is still in use`。
- 只停一個服務、其他繼續跑：`docker compose --profile joycaption stop joycaption`。要再啟動用 `docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile joycaption up -d joycaption`。只想釋放 JoyCaption 的 VRAM、不停容器，見[釋放 VRAM](#釋放-vram)。
- **不要加 `-v`**：那會刪掉 `models`、`ollama`、`hf-cache` 這些 volume，WD14、Ollama 模型和 JoyCaption（約 17 GB）都得重新下載。`./data` 裡的專案不受 `down` 影響。
- `.env` 有設定 `COMPOSE_PROFILES` 的話，直接 `docker compose down` 就會包含那些 profile。

---

## 使用流程

1. **建立專案**：選擇目標底模、LoRA 類型（角色 / 畫風 / 概念）、trigger word。設定會自動套用該底模的建議值。
2. **上傳**：拖放圖片或整個資料夾、zip 皆可；同名 `.txt` 會被當作既有 caption 匯入（方便修改舊資料集）。
   大量圖片可放到 `./import/<資料夾>`，再用「更多匯入 → 從伺服器資料夾匯入」。
3. **標註**：按「▶ 標註未完成」。WD14 產生標籤，natural / hybrid 模式再由 VLM 產生描述。
4. **檢查與修正**
   - 點圖片開啟編輯器：× 刪除標籤、拖曳排序、Enter 新增、← → 換張，自動儲存。
   - 「標籤統計」：看全資料集的標籤頻率，一鍵全部移除 / 取代 / 加入黑名單。
   - Shift / Ctrl + 點擊多選 → 批次加標籤、移除、重新標註。
   - 修改門檻或修剪群組後，用「套用到現有 ▾ → 重新套用門檻 + 修剪 + 黑名單」直接從 WD14 原始分數重建標籤（不需重新推論）。
5. **匯出**：「⬇ 匯出資料集」→ Civitai zip。到 Civitai → Train a LoRA → Dataset 上傳 zip，caption 會一併帶入，底模選擇匯出視窗提示的選項。
   也可匯出 kohya_ss 資料夾結構（`img/10_trigger class/`）或 Hugging Face `metadata.jsonl`。

---

## 各底模的標註方式

| 底模 | Caption 風格 | 特殊標籤 | Clip skip |
|---|---|---|---|
| SD 1.5 動漫 | Danbooru 標籤，≤75 tokens | NSFW 圖加 `nsfw` | 2 |
| SD 1.5 寫實 | 短句 + 關鍵標籤（`ohwx woman, a photo of …`） | `nsfw` | 1 |
| SDXL / 寫實 XL | 1–2 句自然語言 + 少量標籤 | `nsfw` | – |
| Pony V6 XL | Danbooru 標籤 | `source_anime`、`rating_safe/questionable/explicit`；生成時 `score_9, score_8_up, …` | 2 |
| Illustrious XL | Danbooru 標籤：人數 → 角色 → 作品 → 畫師 → 一般 | `general/sensitive/nsfw/explicit`（選用） | 2 |
| NoobAI-XL | 同 Illustrious，可含 e621 標籤 | `general/sensitive/nsfw/explicit` | 2 |
| Animagine XL 3.1 / 4.0 | `1girl, 角色, 作品, rating, 其餘` | `safe/sensitive/nsfw/explicit` | – |
| Anima（Base / Aesthetic / Turbo） | Danbooru 標籤（小寫、空格；也可句子或混合）：分級 → 人數 → 角色 → 作品 → @畫師 → 一般 | `safe/sensitive/nsfw/explicit`（放最前面）；畫師要加 `@` | – |
| FLUX.1 dev | 詳細自然語言（2–5 句），trigger 當主詞，不打亂 | – | – |

完整說明（範例 caption、Civitai 訓練參數、A1111 生成 prompt 範本、NSFW 注意事項）在 WebUI 的「底模標註指南」，或 `GET /api/profiles/<key>/guide.md`。

> **Anima**：原版 A1111 無法使用，生成請用 ComfyUI 或 Forge Neo。訓練請用 Anima-Base，搭配 sd-scripts（`anima_train_network.py`，匯出選 kohya 格式）或 diffusion-pipe；Civitai 上傳時底模選「Anima」。sd-scripts 的範例使用 `--cache_text_encoder_outputs`，這時不能 `--shuffle_caption`，所以此底模的建議是不打亂 caption。

**LoRA 類型與修剪**

- **角色**：預設移除髮色、瞳色標籤，讓 trigger 學會這些固定特徵；服裝、表情、動作保留。
- **畫風**：移除畫風 / 媒材標籤（anime coloring、sketch、watercolor…）與角色名，只描述內容。
- **概念**：保留全部，手動刪除描述該概念本身的標籤。

---

## NSFW

- WD14 直接輸出 Danbooru 的 NSFW 標籤，不做過濾；分級（general / sensitive / questionable / explicit）會依底模換算成 `rating_explicit`、`nsfw`、`explicit` 等寫法（「加入分級標籤」開關）。
- 想做成 SFW 版本：在「修剪 / 黑名單」勾選「NSFW 標籤」修剪群組。有碼 / 無碼標籤（`censored`、`mosaic censoring`、`uncensored`）預設保留，生成時才能控制。
- 自然語言描述：勾選「NSFW 露骨描述」並使用無審查 VLM。建議 `--profile joycaption`（JoyCaption Beta One，8B，bf16 約需 17GB VRAM），`.env` 設定：
  ```
  VLM_BACKEND=openai
  VLM_BASE_URL=http://joycaption:8000/v1   # compose 服務名稱 + vLLM 埠
  VLM_MODEL=joycaption                     # 對應 --served-model-name
  VLM_API_KEY=EMPTY                        # vLLM 預設不驗證，填任意值
  ```
  並把 Ollama 那三行註解掉。第一次啟動會下載約 17GB 模型，`docker compose --profile joycaption logs -f joycaption` 出現 `Application startup complete` 才可使用；右上角 VLM 狀態會轉為可用。
  JoyCaption 服務由 `docker/joycaption.Dockerfile` 建置：以 `vllm/vllm-openai:v0.30.0` 為基底並固定 transformers 5.16.1（0.30.0 內附的 5.17 會讓 Llava 模型無法載入，見 vllm-project/vllm#58755）。
  Claude 與多數商用模型會拒絕描述露骨內容；被拒答時該圖會標記錯誤並保留 WD14 標籤。
- WebUI 預設模糊 NSFW / 露骨縮圖（工具列的「模糊 NSFW」可關閉）。
- 依 Civitai 政策，**未成年特徵標籤（loli、shota、child…）與性內容同時出現的圖片會被標記並自動排除匯出**。

## 白色色塊（遮擋其他人的白色矩形）

為了讓畫面只剩一個人而用白色矩形蓋掉其他人時，caption 沒寫到的色塊會被學進 LoRA，生成時也會畫出來。

- **偵測**：匯入時自動偵測；舊專案用圖庫工具列的「▭ 白色色塊」→「偵測白色色塊」補偵測一次。有色塊的圖片卡片右上角會顯示 ▭。
- **一鍵加入關鍵字**：同一個選單的「一鍵加入關鍵字」，或設定 →「標籤格式 / 分級」的「有白色色塊的圖片加上關鍵字」。有色塊的圖片會在 caption 最後加上關鍵字（預設 `white rectangle`，可修改）；因為是在組 caption 時加上，重新標註也不會消失，關閉後立即恢復。訓練完成後，生成時把這個關鍵字放進負面提示即可去掉色塊。
- **找出、批次處理**：「只顯示有色塊的圖」、「選取有色塊的圖」（再用批次工具列加標籤或刪除）、「刪除有色塊的圖」。
- **誤判修正**：編輯器上方的「▭ 有白色色塊／沒有白色色塊」按鈕可手動標記，↺ 改回偵測結果；重新偵測會保留手動標記。
- 送 Civitai 訓練時，範例提示會去掉這個關鍵字，並把它加進範例的負面提示。
- 偵測方式：找出幾乎純白（RGB ≥ 248）、面積至少 4% 的矩形；有紋理的白牆、過曝窗戶通常不會被當成色塊。
- API：`POST /api/projects/{id}/blocks/scan`；MCP：`detect_white_blocks`；設定 `block_tag_auto` / `block_tag`。

## 放大與降噪（waifu2x）

短邊低於訓練解析度（SDXL / Illustrious 約 1024 px）的圖，訓練時會被一般演算法放大而變糊，LoRA 會把模糊學進去；JPEG 的壓縮雜訊也一樣會被學進去。圖庫工具列的「⤢ 放大 / 降噪」用 waifu2x 處理這兩種情況，對話框最上面選「放大低解析圖」或「只降噪（不改尺寸）」。按鈕上的數字是低於放大門檻的張數；處理過的圖，卡片右上角會顯示 ⤢2x / ⤢4x 或 ✧ 降噪。

- **處理哪些圖**：放大是短邊低於 1024 px（可調整）、還沒處理過的圖；只降噪是還沒處理過的 JPEG / 有損 WebP，不論大小；選「所有還沒處理過的圖（含 PNG）」會連 PNG 一起處理。PNG 存檔本身不產生雜訊，但內容可能帶著影片壓縮或之前 JPEG 留下的雜訊。要自己挑，就選取圖片後按批次列的按鈕，或在編輯器按「放大…」。短邊不到 384 px 的圖會標示「太小」，放大也救不回細節，通常直接刪除比較好。
- **選項**：模型 `art`（插畫 / 動漫，建議）、`art_scan`（有網點、紙紋的掃描圖）、`photo`（照片）。放大時，降噪「自動」只對 JPEG 和有損 WebP 用 1 級，其他不降噪；倍率「自動」用 2x，2x 還不到門檻才用 4x。只降噪時，一般 JPEG 用 1 級就夠；2–3 級只用在方塊雜訊明顯的圖，太強會連細線和紋理一起抹平。
- **原圖會保留**：結果存成 PNG，原圖移到 `data/projects/<id>/originals/`。編輯器或批次列的「還原原圖」、對話框裡的「全部還原原圖」可以換回去。重新放大一律從原圖開始；再次匯入原圖會被當成重複的圖片。
- 標籤不必重跑（WD14 本來就把圖縮到 448 px 判讀）。白色色塊會重新偵測，手動標記會保留。放大後的檔案內容雜湊不同，下次送 Civitai 訓練會上傳新的版本。
- **模型**：[nunif](https://github.com/nagadomi/nunif) 的 waifu2x `swin_unet` ONNX 模型（作者 nagadomi，MIT 授權）。第一次使用時，只從 release 的 zip 裡讀出這次需要的模型（每個約 17–19 MB），存到 `models` volume。和 WD14 共用 onnxruntime，不需要 PyTorch 或 Vulkan。
- **速度與 VRAM**：在 RTX 5090 上實測，960×540 → 1920×1080 每張約 0.3 秒；CPU 約 6–9 秒。每個載入的模型約占 0.6–0.8 GB VRAM，另加 CUDA 本身約 0.5 GB，工作結束時全部自動釋放。放大和標註工作會輪流執行，不會同時跑。
- API：`POST /api/projects/{id}/upscale`（`"scale": 1` = 只降噪）、`POST /api/projects/{id}/upscale/restore`；MCP：`upscale_images`、`restore_upscaled_images`。

## SMB 伺服器（NAS / 分享資料夾）

NAS 或 Windows 分享資料夾裡的圖片，可以由伺服器直接讀取，不經過瀏覽器：專案用「更多匯入 → SMB 伺服器」，角色篩選用「加入 → SMB 伺服器」。

- **連線設定**：
  - 主機（IP 或名稱；貼上 `\\nas\share` 會自動填好分享名稱）、分享名稱、使用者、密碼，網域與連接埠（445）可不填。
  - 「測試連線」會確認主機、帳密與分享都正確，連得上才會儲存。
  - 密碼存在伺服器（`data/studio.db`），沒有加密，和 `.env` 裡的 Civitai 金鑰同一層級；網頁和 API 都不會顯示。編輯時密碼欄留空 = 不變。
- **像檔案瀏覽器一樣挑選**：
  - 點資料夾打開，圖片會顯示縮圖；縮圖可以關掉，改成清單，網路慢時比較快。
  - 勾選資料夾或圖片可以一次匯入多個。換資料夾、換連線時勾選都會保留，所以一次匯入可以來自不同的 NAS。
  - Shift 可以連續選取，「全選這一層」一次勾選目前資料夾的全部項目。
  - ⤢ 放大預覽，← → 切換，空白鍵勾選。
  - 沒有勾選時，匯入目前的資料夾。
  - 勾選的資料夾可以選擇是否包含子資料夾；重複的圖片（例如勾了資料夾又勾了裡面的圖）只會匯入一次。
  - 匯入在背景工作裡執行並顯示進度，大量圖片也不會讓網頁卡住。某個項目讀不到時，其他照常匯入，錯誤列在工作結果裡。
  - 縮圖快取在 `data/cache/smb/`，刪除或修改連線時會清掉。
- **專案**會把圖片下載下來（訓練、匯出都要本機檔案），同名的 `.txt` 會當成既有的 caption（單獨勾選的圖片也會帶上同一資料夾裡的同名 `.txt`）。
- **角色篩選**只記 SMB 路徑，辨識、預覽、下載時才讀取，大量圖片也不多佔空間。刪除連線後這些圖片就讀不到了，刪除前的確認視窗會顯示有幾張。
- app 的容器要連得到 SMB 伺服器。`nas.local` 這類名稱（mDNS）在 Docker 裡可能解析不到，請改用 IP。app 開放給其他電腦連線時，記得設定 `API_KEY`。
- 使用 [smbprotocol](https://github.com/jborean93/smbprotocol)（MIT），支援 SMB 2 / 3。
- API：`GET/POST /api/smb`、`PATCH/DELETE /api/smb/{id}`、`POST /api/smb/test`、`GET /api/smb/{id}/browse?path=`（子資料夾與圖片的名稱、大小、修改時間）、`POST /api/smb/import`（`items`：`[{conn_id, path, dir}]`，可跨連線；`target`：`project` / `finder`）。

## 角色篩選（CCIP）

上方的「角色篩選」可以從一堆圖片挑出某個角色。它和專案分開，不會動到專案裡的圖片。

1. **新增篩選**：選 CCIP 預設模型（150 MB，約 2 GB VRAM）或大模型（384 MB，約 3 GB，稍微準一點）。
2. **目標角色的參考圖**：1 張就能用；3–10 張單人、臉清楚、不同角度與服裝會更穩。和其他參考圖不像的會標「?」，可能是放錯圖。
3. **排除（可不加）**：長得像的其他角色，例如髮色相近的。圖片比較像這些角色時就不算符合。
4. **要篩選的圖片**：多少張都可以。
5. **開始辨識**：結果依相似度排列，符合的會自動勾選。門檻滑桿預設是模型公布的值（0.178 / 0.213），調低比較嚴格。
6. **手動修正**：模型挑錯的按卡片上的 ✕，標為「不是目標」；漏掉的在「不符合」裡按 ✓ 標成「是目標」（下面的 tag 篩選仍然會套用）。放大檢視時可以用 ← → 切換，按 X 排除、V 標成是目標。手動判定會保留，換門檻或重新辨識都不會蓋掉；按 ↺ 取消。
7. **tag 篩選（可不用）**：「同時產生 tag」預設勾選，辨識時用 WD14 幫要篩選的圖片產生 tag。結果上方可以輸入 tag：
   - 必須有某個 tag，或前面加 `-` 代表不能有，例如 `-multiple girls` 去掉多人圖。
   - 「常見」列出這批結果裡出現的 tag 和張數，角色 tag 排在最前面；按 tag 是「必須有」，按旁邊的 − 是「不能有」。
   - 只篩「是目標」的圖片：CCIP 挑出的和手動 ✓ 的。✕ 一定排除。被篩掉的圖片在「被 tag 篩掉」裡，卡片上會寫原因；WD14 標錯時按 ＋（忽略 tag 保留）可以留下來。
   - 有設定篩選時，還沒產生 tag 的圖片無法確認，先不算符合，列在「還沒有 tag」裡；產生 tag 後會自動判斷。所以「符合」就是通過 tag 篩選的目標（加上忽略 tag 保留的）。
   - 篩選條件會存起來，下載和匯入專案都照篩選後的結果。辨識時沒勾的話，結果上方也有「產生 tag」按鈕，只補 tag、不重算特徵。
8. **下載**選取的圖片（zip），或**匯入新的 / 既有的專案**。只匯入圖片，重複的會略過。

- **移除重複**：列出幾乎相同的圖片分組，確認後才處理。有兩個地方：
  - **在篩選結果裡**（結果列的按鈕，或右上角 ⋯ 選單）：只比對目前符合的圖片，重複的改成手動排除（在「不符合」分頁按 ↺ 可以復原）。
  - **在所有要篩選的圖片裡**（「要篩選的圖片」標題列右邊的按鈕）：重複的從這個篩選移除。5 萬張第一次約幾秒，之後調整差異容許幾乎立即更新。
  - **差異容許**（1–16）：兩張圖縮成 32×32 後的平均差異（0–255）。2 以下 = 幾乎一模一樣（尺寸、壓縮、字幕不同）；3–6 = 同一個鏡頭，嘴型、眨眼不同也算；7–11 = 表情、手勢不同也算；更大時鏡頭移動也可能算進來。在所有圖片裡找時，8 以上同背景的不同角色也可能算進來，建議小一點（預設 4；篩選結果裡預設 6）。先用感知雜湊挑出候選。
  - **保留哪張**：優先保留目前符合的，不會留下別的角色那張、丟掉目標角色那張；再來是解析度最高的（同解析度時優先 PNG）。點縮圖可以改留另一張，取消勾選的組不處理。
  - 只從這個篩選移除，SMB、伺服器資料夾和專案裡的原檔不會被刪。辨識過的圖片已經有指紋，可以直接找；還沒辨識的會先讀一次圖片（只用 CPU）。

- **圖片來源**：參考圖、排除參考圖、要篩選的圖片都可以從這些地方加入：
  - 本機的圖片、資料夾、zip（按鈕或拖放）
  - SMB 伺服器（只記路徑，見[SMB 伺服器](#smb-伺服器nas--分享資料夾)）
  - 伺服器匯入資料夾 `./import`（只記路徑，不複製）
  - 既有專案的圖片（用硬連結，不多佔空間）
- 特徵會存起來，所以調門檻是即時的，加入新圖片也只算新的那幾張。參考圖或排除參考圖改變時，特徵可以沿用，但要重新辨識；換模型則全部重算。
- **準確度**：在作者的動畫截圖資料集上測試，包含同系列 5 個長得像的角色和另一部作品的 1 個角色。每個角色只給 1–5 張參考圖，單人圖的辨識正確率是 99.7%；WD14 認不出的圖也能找回約三分之二。
- **限制**：CCIP 是整張圖比對。人物很小、多人同框，或畫面大半被白色色塊蓋住的圖，可能漏掉或認錯。沒有測過原創角色和很新的角色。
- **速度與 VRAM**：RTX 5090 每張約 35 ms（大多花在讀圖），CPU 約 130 ms。工作結束時自動釋放模型；產生 tag 時會暫時載入 WD14，原本沒載入的話結束時一起釋放。和標註、放大共用同一個工作佇列，不會同時跑。「開始辨識」旁邊有和右上角相同的 VRAM 選單；勾選「辨識前先釋放其他模型的 VRAM」，開始前會釋放 WD14、waifu2x，並讓 JoyCaption 休眠。
- **模型**：[deepghs/ccip_onnx](https://huggingface.co/deepghs/ccip_onnx)（OpenRAIL 授權），第一次使用時從 Hugging Face 下載。
- **檔案**：上傳的圖片存在 `data/finder/<id>/`，刪除篩選時一起刪除；伺服器匯入資料夾裡的原檔不會動。
- **API**：`POST /api/finder`、`POST /api/finder/{id}/upload?role=ref|neg|pool`、`/import-server`、`/import-project`、`/run`（`tags: true` 同時產生 tag）、`/manual`、`PATCH /api/finder/{id}`（`tag_filter`）、`GET /duplicates?threshold=1–16&scope=all|matches`、`/hash`、`/download`、`/to-project`，詳見 `/docs`。

## 用 VLM 產生 Danbooru 標籤（JoyCaption）

JoyCaption Beta One 內建「Danbooru tag list」模式。設定面板「Danbooru 標籤（WD14 / VLM）」→「用 VLM 產生 Danbooru 標籤」：

| 模式 | 結果 |
|---|---|
| 關閉（預設） | 只用 WD14 |
| 補充 | WD14 標籤 + VLM 的角色 / 作品 / 畫師標籤 |
| 合併 | WD14 + VLM 的全部標籤 |
| 只用 VLM | 只用 VLM 的標籤，WD14 只負責分級（safe / nsfw…） |

- 每張圖會多一次 VLM 呼叫，與「啟用 VLM 描述」無關；VLM 的原始輸出存在資料庫，調整模式或門檻後用「套用到現有 → 重新套用門檻 + 修剪 + 黑名單」即可，不必再呼叫 VLM。
- **VLM 常編出不存在的角色名與作品名**（實測香風智乃的動畫截圖被判成數個不存在的角色）。WD14 已認出角色、而 VLM 判斷不同時，會自動忽略 VLM 的角色 / 作品標籤；「只用 VLM」模式不做這個檢查。
- 畫師標籤預設不採用（「採用 VLM 判斷的畫師標籤」）；Anima 會自動加上 `@`。
- VLM 標籤沒有信心分數，門檻只套用在 WD14。動漫圖通常仍以 WD14 為主；VLM 標籤適合 WD14 不認得的新角色或冷門概念。
- REST `POST /api/quick-tag(/json)` 與 MCP `quick_tag_image` 可帶 `vlm_tags`；專案設定可用 `update_project_settings` 修改。

## 釋放 VRAM

右上角的「VRAM ▾」可以釋放本機模型占用的 VRAM，需要時再載回。和 ComfyUI 等工具共用顯示卡時很有用。

- **WD14**：「釋放 WD14 的 VRAM」會卸載模型，下次標註時自動載回（約數秒），也可以按「載入 WD14」先載入。CUDA 本身約 0.5 GB 的占用要重啟 app 才會歸零。
- **JoyCaption**：「讓 VLM 休眠」使用 vLLM 的休眠模式，模型權重移到系統記憶體（約 17 GB），不必停容器就能釋放 VRAM。下次呼叫 VLM 時自動喚醒（約數秒），也可以按「喚醒 VLM」。如果這段期間 VRAM 被其他程式占走，喚醒會失敗。
- **waifu2x**：只在放大時載入，工作結束時自動釋放。「釋放 waifu2x 的 VRAM」是給工作中斷時用的。
- 標註或放大工作排隊中或進行中時不能釋放。
- 內建的 `joycaption` 服務以 `--enable-sleep-mode` 和 `VLLM_SERVER_DEV_MODE=1` 啟動 vLLM。容器是在加入這個設定之前建立的話，用 `docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile joycaption up -d joycaption` 重新建立。這會開啟 vLLM 的開發用端點，但這個服務沒有開放埠，只有 app 連得到。
- Ollama 閒置 `OLLAMA_KEEP_ALIVE`（預設 10 分鐘）後會自己卸載模型，這裡不能讓它休眠。
- 有 `nvidia-smi` 時，選單也會顯示整張 GPU 的 VRAM 用量（含其他程式）。
- **CCIP**（角色篩選）：只在辨識時載入，結束時自動釋放。
- API：`GET /api/gpu`、`POST /api/gpu/{release_wd14|load_wd14|sleep_vlm|wake_vlm|release_waifu2x|release_ccip}`。

## 送到 Civitai 雲端訓練

用 Civitai 的 [Orchestration API](https://developer.civitai.com/orchestration/) 在雲端訓練，不必自己準備 GPU：

1. 在 civitai.com → 帳號設定 → API Keys 建立金鑰，填入 `.env` 的 `CIVITAI_API_KEY`，重建 app 容器。金鑰只留在伺服器端，不會送到瀏覽器。
2. 專案頁按「☁ Civitai 訓練」，選擇訓練類型；表單只顯示該類型有的參數，並直接填入該類型的數值（送出的就是畫面上的數字），即時估算 Buzz。
3. 「上傳並試算費用」：圖片以 JPEG 上傳（每張都經過 Civitai 內容審核，被擋下的會列出來），接著用 `whatif` 試算 Buzz，**不會扣款**。
4. 按「確認開始訓練」才會送出並扣 Buzz；之後在「訓練紀錄」看進度、審核結果、每個 epoch 的 LoRA 下載連結與範例圖，也可以取消。

**訓練進度**：首頁上方的「Civitai 雲端訓練」列出所有專案中正在訓練的任務（以及 24 小時內結束的），專案卡片與專案頁的 Civitai 按鈕也會顯示百分比；點進去會直接打開該專案的訓練紀錄。每 15 秒更新：

- 排隊中：前面還有幾個任務、預計開始時間。
- 訓練中：百分比、目前階段（載入底模、訓練、上傳…）、完成的 epoch 與剩下幾個、目前步數與剩下幾步、每步秒數、已進行時間、預估剩餘時間與完成時刻。
- 步數與每步秒數來自 Civitai 的即時追蹤（送出時開啟 `trace: "events"`），剩餘時間 ≈ 剩餘步數 × 每步秒數，epoch 之間的範例圖生成與上傳可能再多一點時間；沒有追蹤資料時改用完成的 epoch 數或 Civitai 的估計。
- API：`GET /api/civitai/active`；MCP：`list_active_civitai_trainings`，`get_civitai_training` 也包含 `progress`。

**訓練類型**：Civitai AI Toolkit 支援、可以用圖片資料集訓練的全部 26 種。

| 分類 | 訓練類型 |
|---|---|
| 圖片 | SD 1.5、SDXL、Anima、Flux.1 dev / schnell、Flux.2 Klein 4B / 9B、Chroma1-HD、ERNIE-Image、Qwen-Image（latest / 2509）、Qwen Image 2.1、Z-Image Turbo / Base、Boogu、HiDream O1、Ideogram 4、Krea 2、Mage-Flow、Ming |
| 影片（可用圖片訓練） | LTX-2、LTX-2.3、LTX-2.5、Wan 2.1 / 2.2（Civitai 標示為預覽版）、MiniMax H3 |

- 預設依專案底模選擇：SD 1.5 → `sd1`；SDXL / Pony / Illustrious / NoobAI / Animagine → `sdxl`，並分別用 SDXL 1.0、Pony Diffusion V6 XL、Illustrious-XL v0.1、NoobAI-XL eps 1.1、Animagine XL 4.0 當訓練底模；Anima → `anima`（Anima-Base v1.0）；FLUX.1 dev → `flux1-dev`。
- 數值與價格取自 [Civitai 文件](https://developer.civitai.com/orchestration/recipes/)；文件沒列出的類型（Anima、Qwen 2.1、Boogu、HiDream O1、Ideogram 4、Krea 2、Mage-Flow、Ming、LTX-2.5、MiniMax H3）填入 AI Toolkit 的通用數值，費用以試算為準。
- 音樂類型（ACE-Step、YuE2）需要音訊資料，這個工具只處理圖片，不支援。

**參數欄位**（依 Civitai 規格 `v2-consumers.json`，各類型不同）：

| 參數 | 適用類型 |
|---|---|
| 步數、epochs、學習率、學習率排程、optimizer、network dim / alpha、noise offset、flip、shuffle / keep tokens、接續訓練的 LoRA、範例提示 / 負面提示 / CFG / LoRA 強度 | 全部 |
| batch size | 上限 SD 1.5 / SDXL 4；Flux.2 Klein 4B、ERNIE、Z-Image 2；其他固定 1 |
| 自訂訓練底模 | 只有 SD 1.5、SDXL、Anima（其他類型的底模由 Civitai 固定） |
| trigger word | SD 1.5、SDXL、Flux.1、Flux.2 Klein、Chroma、Z-Image（其他類型 trigger 寫在每張 caption 開頭） |
| Min SNR γ、同時訓練文字編碼器、文字編碼器學習率 | 只有 SD 1.5、SDXL（其他類型文件寫明不訓練文字編碼器） |

- 預設值：學習率排程 cosine；noise offset SDXL 0.1、其他 0；Min SNR γ 5；文字編碼器學習率 5e-5；範例 LoRA 強度 1.0；範例 CFG 用 Civitai 生成 API 對該底模的預設（SDXL 7、Flux.1 3.5、Klein 5、Qwen 2.5、Anima 4…），Flux.1 schnell 與 MiniMax H3 沒有公開數值，留空由 Civitai 決定。
- 訓練底模與接續訓練的 LoRA 可填 AIR、模型版本 ID 或含 `modelVersionId` 的 civitai.com 網址（用 Site API 解析）。會檢查底模的類型是否相符；LoRA 會檢查是不是 LoRA，以及 SD / Flux / Klein / Chroma / ERNIE / Qwen / Z-Image / Anima 是否同一種底模（Wan / LTX 在網站上的名稱不同，交給 Civitai 試算時驗證）。
- Civitai 網站訓練器的 resolution、repeats、clip skip 等設定不在 API 裡，無法指定；訓練解析度由 Civitai 依底模決定。Flux.2 Klein 的編輯訓練（`isEditTraining`）需要成對的參考圖，不支援。
- 只上傳匯出時也會包含的圖片（依 Civitai 政策排除「未成年特徵 + 性內容」）；caption 與匯出的 `.txt` 相同，shuffle / keep tokens 預設依標註設定帶入。Civitai 每張 caption 最多 1024 字：超過時從 WD14 分數最低的標籤開始拿掉，直到放得下（自己加的標籤最後才拿），trigger、描述、附加標籤和色塊關鍵字一律保留。

**上傳與重新訓練**

- 已上傳的圖片以圖片內容辨識，會記住約 25 天，重新送訓練時不必再上傳。caption 是隨訓練請求送出的，所以**只改標籤不需要重傳**；換了圖片、改了「上傳圖片長邊」、換了金鑰才會重傳，Civitai 不認得時也會自動重傳。
- 「強制重新上傳所有圖片」（API / MCP：`force_upload`）會全部重傳，例如圖片檔在系統外被改過、或想讓 Civitai 重新審核時。
- **用相同參數重新訓練**：訓練紀錄每一筆都有這個按鈕，會把那次的訓練參數、範例提示與上傳設定填回表單（標籤用目前的；排隊優先度維持表單的選擇），一樣要試算、確認後才送出。API / MCP 的訓練狀態裡 `request` 就是那次的參數。

**排隊優先度**：「一般」等於 Civitai 網站訓練器的 High Priority 開關（這裡的預設）；「低」是網站沒開時的值，API 不指定時也是低；「高」只有 API 能選，效果依帳號等級。實測試算的價格三者相同，實際價格以試算為準。

**費用、取消與退款**

- 試算可能已套用 Civitai 的折扣，這時表單會同時顯示折扣前的原價：實際扣款可能以原價結算（送出時預扣、結束時追加）。訓練紀錄顯示的是 Civitai 交易紀錄的實際扣款。
- 取消訓練送出的是 `status: canceled`（與 Civitai 網站相同），訓練會停止且無法接續。取消是非同步的：Civitai 要等訓練機器停下來才會改成「已取消」（可能要幾分鐘），這段時間會顯示「取消中」。Civitai 文件只說明還沒開始的任務可能退款；取消後訓練紀錄會顯示實際退回的 Buzz（結束後一小時內持續查詢）。
- 訓練費用依 Civitai 當下的價格（例如 SDXL / SD1 每步 0.2 Buzz + 每個 epoch 10 Buzz，且不低於預設配置的 80%），以試算結果為準。

**匯入 A1111 / Forge**：訓練紀錄的每個 epoch 旁邊有「→ A1111」按鈕，按了才匯入。
- 需要 [Forge Neo Chino](https://github.com/sss22213/sd-webui-forge-neo-chino) 的 LoRA 匯入 API（`POST /sdapi/v1/lora/import`），並在 `.env` 設定 `A1111_URL`。Forge 跑在同一台主機時用 `http://host.docker.internal:7860`；沒設定就不顯示這個按鈕。
- Forge 會自己從 Civitai 的簽名網址下載 LoRA，不經過這裡。檔案放在 Lora 資料夾的 `A1111_LORA_SUBFOLDER`（預設 `LoRA-Tag-Studio`），檔名像 `kirima_syaro_20261004_3rmo_e10.safetensors`（觸發詞、訓練日期與編號、epoch）。
- 卡片資訊會一起寫好：觸發詞（點卡片時自動加在 `<lora:…>` 後面）、底模類型（Forge 依預設篩選 LoRA 用）、說明（專案與 epoch），第一張範例圖當預覽。
- 不必重開 Forge，在 LoRA 分頁按重新整理就看得到。完成時會顯示可以直接貼上的提示詞。
- Forge 已經有同一個檔案時不重複下載。同名但內容不同時會先問要不要覆蓋。已匯入的 epoch 會標示「已在 A1111」。
- API：`POST /api/civitai/runs/{id}/a1111`（`epoch`、`overwrite`），`GET /api/a1111` 檢查 Forge 連不連得上；MCP 工具 `import_lora_to_a1111`。

**API / MCP**：`GET /api/civitai` 列出每個類型的 `fields`、`defaults`、`max_batch`；送出該類型沒有的參數會回錯誤。MCP 流程：`get_civitai_training_types` → `prepare_civitai_training`（可帶 `training_type`、`priority`、`force_upload` 與上表的參數）→ `get_civitai_preparation`（取得試算費用）→ 使用者確認費用後才 `submit_civitai_training` → `get_civitai_training`。

---

## 給其他 LLM 使用

WebUI 的「API / MCP」頁面有可直接複製的設定。

### MCP

Streamable HTTP 端點：`http://<host>:7870/mcp`

```bash
# Claude Code
claude mcp add --transport http lora-tag-studio http://localhost:7870/mcp
```

```jsonc
// Cursor / VS Code / 其他支援 HTTP 的客戶端
{ "mcpServers": { "lora-tag-studio": { "type": "http", "url": "http://localhost:7870/mcp" } } }
```

共 27 個工具：

| 類別 | 工具 |
|---|---|
| 底模與快速標註 | `list_profiles`、`get_tagging_guide`、`quick_tag_image` |
| 專案與匯入 | `list_projects`、`create_project`、`get_project`、`update_project_settings`、`add_images_from_urls`、`list_server_import_folders`、`import_server_folder` |
| 標註與編輯 | `start_tagging`、`get_job_status`、`list_captions`、`get_image`、`update_image_caption`、`bulk_edit_tags`、`get_tag_stats`、`detect_white_blocks` |
| 放大 | `upscale_images`、`restore_upscaled_images` |
| 匯出 | `export_dataset` |
| Civitai 訓練 | `get_civitai_training_types`、`prepare_civitai_training`、`get_civitai_preparation`、`submit_civitai_training`、`get_civitai_training`、`list_active_civitai_trainings` |

### REST / OpenAPI

- Swagger UI：`/docs`，規格：`/openapi.json`（可直接加到 Open WebUI 的 OpenAPI Tool Server、GPTs Actions 等）
- 給 LLM 閱讀的說明：`/llms.txt`

```bash
# 單張圖片直接產生 caption（不建專案）
curl -X POST http://localhost:7870/api/quick-tag/json -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/a.png","profile":"pony_v6","trigger":"mychar"}'

# 完整流程
curl -X POST http://localhost:7870/api/projects -H "Content-Type: application/json" \
  -d '{"name":"my-char","profile":"illustrious","lora_type":"character","trigger":"mychar"}'
curl -X POST http://localhost:7870/api/projects/<id>/upload -F "files=@dataset.zip"
curl -X POST http://localhost:7870/api/projects/<id>/tag -H "Content-Type: application/json" -d '{"only_untagged":true}'
curl http://localhost:7870/api/jobs/<job_id>
curl -OJ "http://localhost:7870/api/projects/<id>/export?format=civitai"
```

設定 `API_KEY` 後，所有 `/api` 與 `/mcp` 請求需帶 `Authorization: Bearer <API_KEY>`（或 `X-API-Key`）；WebUI 會提示輸入並存在 cookie。

**回應語言**：API 的訊息、底模名稱與指南依序採用 `?lang=`、`X-Lang` header、`Accept-Language`，都沒有時用 `DEFAULT_LANG`（預設 `en`）。MCP 使用 `DEFAULT_LANG`，`get_tagging_guide` 另可傳 `language`。可用代碼：`zh-TW`、`en`、`ja`、`ko`、`zh-CN`（`GET /api/i18n` 列出）。

---

## 設定（.env）

| 變數 | 預設 | 說明 |
|---|---|---|
| `APP_PORT` | `7870` | 對外埠（避開 A1111 的 7860） |
| `API_KEY` | 空 | 設定後啟用驗證，對外開放時務必設定 |
| `PUBLIC_BASE_URL` | 空 | 回傳給 LLM 的下載連結前綴，例如 `http://192.168.1.10:7870` |
| `DEFAULT_LANG` | `en` | API / MCP 未指定語言時的語言：`zh-TW` / `en` / `ja` / `ko` / `zh-CN`（WebUI 依瀏覽器自動選擇） |
| `WD14_MODEL` | `SmilingWolf/wd-eva02-large-tagger-v3` | 預設 WD14 模型（專案可個別覆寫） |
| `ORT_DEVICE` | `auto` | `auto` / `cpu` / `cuda`（waifu2x 也用這個設定） |
| `WAIFU2X_DIR` | `/models/waifu2x` | waifu2x 模型的存放位置 |
| `WAIFU2X_MODELS_URL` | nunif release 的 `waifu2x_onnx_models_20250502.zip` | 讀取 waifu2x 模型的 zip（伺服器需支援分段下載） |
| `TAG_CONCURRENCY` | `2` | 同時處理的圖片數（使用遠端 VLM 時可調高） |
| `HF_TOKEN` | 空 | 選填的 Hugging Face token（提高下載速率） |
| `VLM_BACKEND` | `openai` | `openai`（OpenAI 相容）/ `anthropic` / `none` |
| `VLM_BASE_URL` / `VLM_MODEL` / `VLM_API_KEY` | Ollama | OpenAI 相容端點設定 |
| `VLM_TIMEOUT` | `180` | VLM 請求逾時秒數 |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` / `ANTHROPIC_EFFORT` | 空 / `claude-opus-5-5` / `low` | `VLM_BACKEND=anthropic` 時使用（已啟用伺服器端 refusal fallback） |
| `ALLOW_PRIVATE_URLS` | `0` | 「從網址匯入」是否允許內網位址（預設關閉以避免 SSRF） |
| `CIVITAI_API_KEY` | 空 | Civitai 雲端訓練用的金鑰（只留在伺服器端） |
| `A1111_URL` | 空 | A1111 / Forge 的網址（需要 Forge Neo Chino 的 LoRA 匯入 API），例如 `http://host.docker.internal:7860`；設定後訓練紀錄才有「→ A1111」 |
| `A1111_LORA_SUBFOLDER` | `LoRA-Tag-Studio` | 匯入到 Lora 資料夾裡的哪個子資料夾（空白 = Lora 資料夾本身） |
| `A1111_API_AUTH` | 空 | Forge 有開 `--api-auth` 時填 `帳號:密碼` |
| `JOYCAPTION_GPU_UTIL` | `0.7` | vLLM 預先佔用的 VRAM 比例 |
| `JOYCAPTION_VLLM_IMAGE` / `JOYCAPTION_TRANSFORMERS` | `vllm/vllm-openai:v0.30.0` / `5.16.1` | JoyCaption 的 vLLM 映像檔與 transformers 版本；修改後需 `--build` |

資料位置：`./data`（SQLite、圖片、縮圖、匯出 zip），`./import`（唯讀的伺服器匯入資料夾）。兩者都不會進 git。

---

## 專案結構

```
app/
  main.py            FastAPI 進入點、API 金鑰驗證、語言協商、MCP 掛載、/llms.txt
  i18n.py            語系檔載入、語言協商、翻譯
  locales/*.json     各語言文字（server：後端 / 指南，ui：WebUI）  ← 翻譯改這裡
  api.py             REST API
  mcp_server.py      MCP 工具
  profiles.py        各底模的預設值、分級對應、標註指南   ← 要新增 / 調整底模改這裡
  pipeline.py        單張圖片標註流程
  jobs.py            背景工作佇列（標註、放大）
  services.py        API / MCP 共用邏輯
  exporter.py        Civitai / kohya / jsonl 匯出
  civitai.py         送到 Civitai 雲端訓練（Orchestration API）
  a1111.py           訓練好的 LoRA 匯入 A1111 / Forge
  upscale.py         waifu2x 放大：模型下載、切塊推論、原圖備份 / 還原
  finder.py          角色篩選：CCIP 特徵、比對分數、tag 篩選、找重複、圖片來源、下載 / 匯入專案
  smb.py             SMB 來源：連線設定、瀏覽與縮圖、匯入勾選的資料夾 / 圖片
  storage.py         上傳、資料夾 / zip 匯入、縮圖
  db.py              SQLite
  tagging/
    wd14.py          WD14 ONNX 推論
    vlm.py           OpenAI 相容 / Claude 自然語言描述
    postprocess.py   門檻、排序、修剪群組、黑名單、caption 組合
    blocks.py        白色色塊（遮擋用的白色矩形）偵測
web/                 WebUI（純 HTML/CSS/JS，無需建置）
docker/
  joycaption.Dockerfile  JoyCaption 用的 vLLM 映像檔
scripts/
  check_locales.py   檢查各語系檔的鍵與 {佔位符} 是否與 en.json 一致
```

## 開發

本機執行（不用 Docker，Python 3.12）：

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt onnxruntime
DATA_DIR=./data IMPORT_DIR=./import VLM_BACKEND=none uvicorn app.main:app --reload --port 7870
```

**修改或新增語言**：編輯 `app/locales/<代碼>.json`（缺少的鍵會退回英文），新增語言只要放一個新的 JSON，`_meta.order` 決定選單順序。改完執行 `python3 scripts/check_locales.py`（只需標準函式庫），再重新 `--build`。

---

## 疑難排解

- **VLM 顯示無法連線**：確認有加 `--profile vlm`，且 `docker compose logs ollama-pull` 顯示模型已下載完成。
- **VLM 回 404**：模型名稱與 `ollama list` 不一致。
- **GPU 沒有被使用**：右上角顯示「WD14 · CPU」時，確認使用了 `docker-compose.gpu.yml` 並重新 `--build`，以及 `docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi` 能正常執行。
- **JoyCaption 記憶體不足**：調低 `JOYCAPTION_GPU_UTIL` 或改用較小的 VLM；WD14 GPU 版約佔 1–2GB。和其他工具共用顯示卡時，不標註的時候可以從「VRAM ▾」讓 JoyCaption 休眠。
- **VRAM 選單說 VLM 不能休眠**：joycaption 容器建立時還沒有休眠模式，用 `docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile joycaption up -d joycaption` 重新建立。
- **`docker compose down` 出現 `Network lora-tag-studio_default  Resource is still in use`**：還有 profile 裡的容器（JoyCaption 或 Ollama）在跑。改用 `docker compose --profile vlm --profile joycaption down`（見[停止](#停止)）。
- **JoyCaption 啟動失敗 `cannot import name 'PixtralRotaryEmbedding'`**：使用了未修正的 `vllm/vllm-openai` 映像檔。確認 `docker-compose.yml` 的 joycaption 使用 `build:`，並執行 `docker compose ... --profile joycaption up -d --build`。
- **WebUI 顯示 `common.server_unreachable` 之類的鍵名**：映像檔內沒有語系檔（`app/locales/*.json`），重新 `--build`；`docker compose logs app` 啟動時會列出已載入的語言。
- **放大卡在「準備 waifu2x 模型」就失敗**：app 第一次要連到 github.com 下載模型，錯誤原因可以看 `docker compose logs app`。
- **caption 超過 75 tokens**：SD1.5 / SDXL 的 CLIP 一次讀 75 tokens，降低「最多標籤數」或提高門檻；kohya 可用 `--max_token_length=225`。

## 注意事項

- 本專案與 Civitai 沒有關係。雲端訓練會扣你自己 Civitai 帳號的 Buzz，確認前請先看試算。
- 請確認你有權使用拿來訓練的圖片，並遵守所用服務的內容規範。
- 模型（WD14、waifu2x、CCIP、JoyCaption、Ollama 的模型）在執行時從原作者處下載，適用各自的授權。

## 授權

[MIT](LICENSE)
