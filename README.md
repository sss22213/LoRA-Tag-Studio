# LoRA Tag Studio

**English** | [繁體中文](README.zh-TW.md)

A self-hosted tool that turns a folder of images into a LoRA training dataset. It captions every image automatically (WD14 Danbooru tags plus natural-language descriptions from a vision-language model), lets you review and edit the captions in a web UI, and exports a dataset that Civitai, kohya_ss or Hugging Face can train on directly. It can also send the dataset straight to Civitai's cloud trainer.

- **Captions that match the base model**: SD 1.5 (anime / realistic), SDXL, Pony V6, Illustrious, NoobAI, Animagine, Anima and FLUX.1 each get their own caption format, tag order, rating tags, quality tags and training advice.
- **Two taggers**: WD14 (SmilingWolf v3, ONNX, CPU or GPU) for Danbooru tags, and a VLM (Ollama, JoyCaption, Claude or any OpenAI-compatible endpoint) for natural-language descriptions. JoyCaption's Danbooru mode can also add to or replace the WD14 tags.
- **NSFW support**: WD14 outputs NSFW tags unfiltered, rating tags are converted per base model, there is an explicit-description mode for VLMs, and an optional uncensored JoyCaption service.
- **Web UI**: drag-and-drop folders or zips, gallery filters, a tag editor (drag to reorder, autocomplete), tag statistics, bulk add / remove / replace, trait pruning for character LoRAs, and a blacklist.
- **waifu2x upscaling and denoising**: images whose short side is too small are upscaled before training, so the LoRA does not learn blur, and JPEG noise can be removed without changing the size. Originals are kept and can be restored.
- **Civitai cloud training**: upload images and captions, get a Buzz estimate, start training after you confirm, follow the progress, and download the LoRA of every epoch (Civitai Orchestration API).
- **Usable by other LLMs**: a REST API (its OpenAPI spec works as a tool server) and an MCP server (Claude Desktop, Claude Code, Cursor, Open WebUI).
- **Five languages**: the web UI, error messages, tagging guides and exported READMEs are available in English, 繁體中文, 日本語, 한국어 and 简体中文. The UI follows the browser language and can be switched in the top-right corner.

---

## Requirements

- Docker with Docker Compose v2.
- Optional NVIDIA GPU: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) and driver ≥ 580 (onnxruntime-gpu 1.30 is built for CUDA 13; RTX 50 series is supported).
- Disk space for models, downloaded on first use: WD14 eva02-large about 1.2 GB, Ollama `qwen2.5vl:7b` about 6 GB, JoyCaption about 17 GB.
- JoyCaption needs about 17 GB of VRAM (bf16).

## Quick start

```bash
git clone https://github.com/sss22213/LoRA-Tag-Studio.git
cd LoRA-Tag-Studio
cp .env.example .env          # edit as needed; every variable is described under Configuration
docker compose up -d --build  # CPU only
```

Open <http://localhost:7870>. The WD14 model is downloaded the first time you tag images and kept in the `models` volume.

| Setup | Command |
|---|---|
| CPU, WD14 tags only | `docker compose up -d --build` |
| CPU + Ollama descriptions | `docker compose --profile vlm up -d --build` |
| NVIDIA GPU + Ollama | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile vlm up -d --build` |
| NVIDIA GPU + JoyCaption (NSFW) | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile joycaption up -d --build`, and change the three `VLM_*` lines in `.env` (see [NSFW](#nsfw)) |

> You can also set `COMPOSE_FILE` / `COMPOSE_PROFILES` in `.env`; after that, `docker compose up -d` is enough.

`--profile vlm` starts Ollama and downloads `VLM_MODEL` (default `qwen2.5vl:7b`). To use another model:

```bash
docker compose exec ollama ollama pull qwen2.5vl:32b   # then set VLM_MODEL in .env to the same name
```

### Stopping

```bash
docker compose --profile vlm --profile joycaption down
```

- Include the profiles even if you only started one of them; naming a profile that is not running does no harm. Without them, `down` skips the Ollama / JoyCaption containers and stops with `Network lora-tag-studio_default  Resource is still in use`.
- To stop only one service and keep the rest running: `docker compose --profile joycaption stop joycaption`. Start it again with `docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile joycaption up -d joycaption`. To free JoyCaption's VRAM without stopping it, see [Freeing VRAM](#freeing-vram).
- **Do not add `-v`**: it deletes the `models`, `ollama` and `hf-cache` volumes, so WD14, the Ollama model and JoyCaption (about 17 GB) have to be downloaded again. Your projects in `./data` are not affected by `down`.
- If `.env` sets `COMPOSE_PROFILES`, a plain `docker compose down` covers those profiles.

---

## Workflow

1. **Create a project**: pick the target base model, the LoRA type (character / style / concept) and a trigger word. The recommended settings for that base model are applied automatically.
2. **Upload**: drop images, whole folders or zips. A `.txt` file with the same name as an image is imported as its existing caption, so you can edit an old dataset.
   For large datasets, put the folder in `./import/<folder>` and use "More import → Import from a server folder".
3. **Tag**: click "▶ Tag remaining". WD14 produces the tags; in natural / hybrid mode the VLM then writes a description.
4. **Review and fix**
   - Click an image to open the editor: × removes a tag, drag to reorder, Enter adds a tag, ← → moves between images. Changes are saved automatically.
   - "Tag stats" shows tag frequencies across the dataset; remove, replace or blacklist a tag everywhere in one click.
   - Shift / Ctrl + click selects several images for bulk add, remove or re-tag.
   - After changing thresholds or pruning groups, "Apply to existing ▾ → Re-apply threshold + prune + blacklist" rebuilds the tags from the stored WD14 scores without running the model again.
5. **Export**: "⬇ Export dataset" → Civitai zip. On Civitai, open Train a LoRA → Dataset and upload the zip; the captions come with it. The export dialog tells you which base model to choose.
   kohya_ss folder layout (`img/10_trigger class/`) and Hugging Face `metadata.jsonl` are also available.

---

## Captions by base model

| Base model | Caption style | Special tags | Clip skip |
|---|---|---|---|
| SD 1.5 anime | Danbooru tags, ≤ 75 tokens | `nsfw` on NSFW images | 2 |
| SD 1.5 realistic | A short sentence + key tags (`ohwx woman, a photo of …`) | `nsfw` | 1 |
| SDXL / realistic XL | 1–2 natural-language sentences + a few tags | `nsfw` | – |
| Pony V6 XL | Danbooru tags | `source_anime`, `rating_safe/questionable/explicit`; for generation `score_9, score_8_up, …` | 2 |
| Illustrious XL | Danbooru tags: people count → character → series → artist → general | `general/sensitive/nsfw/explicit` (optional) | 2 |
| NoobAI-XL | Same as Illustrious; e621 tags allowed | `general/sensitive/nsfw/explicit` | 2 |
| Animagine XL 3.1 / 4.0 | `1girl, character, series, rating, everything else` | `safe/sensitive/nsfw/explicit` | – |
| Anima (Base / Aesthetic / Turbo) | Danbooru tags (lowercase, spaces; sentences or a mix also work): rating → people count → character → series → @artist → general | `safe/sensitive/nsfw/explicit` first; artist tags need `@` | – |
| FLUX.1 dev | Detailed natural language (2–5 sentences), trigger word as the subject, no shuffling | – | – |

The full guide for each base model (example captions, Civitai training settings, A1111 prompt templates, NSFW notes) is in the web UI under "Tagging guides", or at `GET /api/profiles/<key>/guide.md`.

> **Anima**: stock A1111 cannot run it; generate with ComfyUI or Forge Neo. Train on Anima-Base with sd-scripts (`anima_train_network.py`, export in kohya format) or diffusion-pipe, or choose "Anima" as the base model on Civitai. The sd-scripts example uses `--cache_text_encoder_outputs`, which cannot be combined with `--shuffle_caption`, so captions are not shuffled for this base model.

**LoRA types and pruning**

- **Character**: hair-color and eye-color tags are removed by default so the trigger word learns these fixed traits; clothing, expressions and poses are kept.
- **Style**: style and medium tags (anime coloring, sketch, watercolor…) and character names are removed, so captions only describe the content.
- **Concept**: everything is kept; remove the tags that describe the concept itself by hand.

---

## NSFW

- WD14 outputs Danbooru's NSFW tags without filtering. The rating (general / sensitive / questionable / explicit) is converted to the base model's format, such as `rating_explicit`, `nsfw` or `explicit` (the "Add rating tags" switch).
- For a SFW version, tick the "NSFW tags" prune group (Settings → Prune / blacklist). Censorship tags (`censored`, `mosaic censoring`, `uncensored`) are kept by default so you can control them when generating.
- Natural-language descriptions: tick "Explicit NSFW descriptions" and use an uncensored VLM. The recommended option is `--profile joycaption` (JoyCaption Beta One, 8B, about 17 GB of VRAM in bf16) with these settings in `.env`:
  ```
  VLM_BACKEND=openai
  VLM_BASE_URL=http://joycaption:8000/v1   # compose service name + vLLM port
  VLM_MODEL=joycaption                     # matches --served-model-name
  VLM_API_KEY=EMPTY                        # vLLM does not check the key by default
  ```
  Comment out the three Ollama lines. The first start downloads about 17 GB; JoyCaption is ready when `docker compose --profile joycaption logs -f joycaption` shows `Application startup complete`, and the VLM status in the top-right corner turns available.
  The JoyCaption image is built from `docker/joycaption.Dockerfile`: `vllm/vllm-openai:v0.30.0` with transformers pinned to 5.16.1, because the bundled 5.17 breaks loading Llava models (vllm-project/vllm#58755).
  Claude and most commercial models refuse to describe explicit content; a refused image is marked with an error and keeps its WD14 tags.
- NSFW and explicit thumbnails are blurred by default ("Blur NSFW" in the toolbar).
- Following Civitai's policy, **images that combine minor-coded tags (loli, shota, child…) with sexual content are flagged and excluded from every export**.

## White masking blocks

If you covered other people with white rectangles so that only one person is left, the rectangles are not in the captions, so the LoRA learns them and draws them when you generate.

- **Detection**: runs automatically on import. For older projects, use "▭ White blocks" → "Detect white blocks" in the gallery toolbar once. Images with blocks show ▭ in the top-right corner of their card.
- **One-click keyword**: "Add “white rectangle” to the … images with blocks" in the same menu, or Settings → Tag format / rating → "Add a keyword to images with white blocks". The keyword (default `white rectangle`, editable) is appended to the caption of those images. It is added when the caption is built, so re-tagging does not remove it and turning it off takes effect immediately. After training, put the keyword in the negative prompt to keep the blocks out of your generations.
- **Find and act on them**: "Show only images with blocks", "Select the … images with blocks" (then use the bulk bar to add tags or delete them), "Delete the … images with blocks".
- **Wrong detection**: the "▭ Has white block / No white block" button at the top of the editor overrides the result, and ↺ goes back to the detected value. Re-running detection keeps manual overrides.
- When training on Civitai, the keyword is removed from the sample prompts and added to the sample negative prompt.
- How it works: it looks for almost pure white (RGB ≥ 248) rectangles covering at least 4% of the image; textured white walls and blown-out windows are usually not detected.
- API: `POST /api/projects/{id}/blocks/scan`; MCP: `detect_white_blocks`; settings `block_tag_auto` / `block_tag`.

## Upscaling and denoising (waifu2x)

Trainers enlarge images whose short side is below the training resolution (about 1024 px for SDXL / Illustrious) with a plain resize. That blurs them, and the LoRA learns the blur. JPEG compression noise is learned too. The "⤢ Upscale / denoise" button in the gallery toolbar fixes both with waifu2x; pick "Upscale low-resolution images" or "Denoise only (keep the size)" at the top of the dialog. The number on the button is how many images are below the upscale threshold; processed images show ⤢2x / ⤢4x or ✧ denoised on their card.

- **Which images**: upscaling picks every image not processed yet whose short side is below 1024 px (you can change the threshold). Denoising picks every JPEG / lossy WebP not processed yet, whatever its size, or with "All images not processed yet (including PNG)" the PNGs too: saving as PNG adds no noise, but a PNG can still carry noise from video compression or an earlier JPEG. To pick images yourself, select them and use the button in the bulk bar, or "Upscale…" in the editor. Images with a short side below 384 px are marked "too small": upscaling cannot bring their detail back, so removing them is usually better.
- **Options**: model `art` (illustrations / anime, recommended), `art_scan` (scans with halftone or paper texture) or `photo`. When upscaling, noise reduction "Auto" uses level 1 for JPEG and lossy WebP and none otherwise, and scale "Auto" uses 2x, or 4x when 2x still stays below the threshold. When only denoising, level 1 suits ordinary JPEGs; levels 2–3 are for clearly blocky images and also smooth away fine lines and textures.
- **Originals are kept**: the result is saved as PNG and the original goes to `data/projects/<id>/originals/`. "Restore original" (editor or bulk bar) or "Restore all originals" (in the dialog) puts it back. Upscaling again always starts from the original, and importing the original again is recognized as a duplicate.
- Tags do not need to be redone (WD14 looks at a 448 px copy anyway). White blocks are re-detected; manual marks are kept. The upscaled file has a new content hash, so the next Civitai training uploads the new version.
- **Models**: the waifu2x `swin_unet` ONNX models from [nunif](https://github.com/nagadomi/nunif) (by nagadomi, MIT). On first use, only the models a job needs are read straight out of the release zip (about 17–19 MB each) into the `models` volume. They run on the same onnxruntime as WD14, so no PyTorch or Vulkan is needed.
- **Speed and VRAM**: measured on an RTX 5090, 960×540 → 1920×1080 takes about 0.3 s per image; on CPU it takes about 6–9 s. Each loaded model uses about 0.6–0.8 GB of VRAM, plus about 0.5 GB for CUDA itself, and everything is freed when the job ends. Upscaling and tagging jobs run one after the other, never at the same time.
- API: `POST /api/projects/{id}/upscale` (`"scale": 1` = denoise only), `POST /api/projects/{id}/upscale/restore`; MCP: `upscale_images`, `restore_upscaled_images`.

## Danbooru tags from a VLM (JoyCaption)

JoyCaption Beta One has a built-in "Danbooru tag list" mode. Settings → "Danbooru tags (WD14 / VLM)" → "Danbooru tags from the VLM":

| Mode | Result |
|---|---|
| Off (default) | WD14 only |
| Supplement | WD14 tags + the VLM's character / series / artist tags |
| Merge | All WD14 tags + all VLM tags |
| VLM only | Only the VLM's tags; WD14 only provides the rating (safe / nsfw…) |

- This adds one VLM call per image, independent of "Enable VLM descriptions". The raw VLM output is stored, so after changing the mode or thresholds, "Apply to existing → Re-apply threshold + prune + blacklist" is enough; the VLM is not called again.
- **VLMs often invent characters and series that do not exist** (in testing, anime screenshots of Chino Kafuu were labeled as several characters that do not exist). When WD14 has recognized a character and the VLM disagrees, the VLM's character and series tags are ignored. "VLM only" mode skips this check.
- Artist tags from the VLM are off by default ("Use artist tags guessed by the VLM"); for Anima the `@` prefix is added automatically.
- VLM tags have no confidence scores, so thresholds only apply to WD14. For anime images WD14 is usually the better source; VLM tags help with new characters or niche concepts WD14 does not know.
- REST `POST /api/quick-tag(/json)` and MCP `quick_tag_image` accept `vlm_tags`; project settings can be changed with `update_project_settings`.

## Freeing VRAM

The "VRAM ▾" button in the top-right corner frees the VRAM held by the local models and loads them back, which helps when the GPU is shared with ComfyUI or other tools.

- **WD14**: "Free WD14's VRAM" unloads the model. The next tagging loads it again automatically (a few seconds), or use "Load WD14" to load it in advance. About 0.5 GB used by CUDA itself stays until the app restarts.
- **JoyCaption**: "Put the VLM to sleep" uses vLLM's sleep mode. The model weights move to system RAM (about 17 GB) and the VRAM is freed without stopping the container. The next VLM call wakes it automatically (a few seconds), or use "Wake the VLM". Waking fails if other programs have taken the VRAM in the meantime.
- **waifu2x**: loaded only while upscaling and freed automatically when the job ends. "Free waifu2x's VRAM" is there in case a job was interrupted.
- Freeing is refused while a tagging or upscaling job is queued or running.
- The bundled `joycaption` service starts vLLM with `--enable-sleep-mode` and `VLLM_SERVER_DEV_MODE=1`. If your container was created before this, recreate it with `docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile joycaption up -d joycaption`. The development endpoints this turns on are only reachable from the app, because the service publishes no port.
- Ollama unloads its model by itself after `OLLAMA_KEEP_ALIVE` (default 10 minutes) and cannot be put to sleep from here.
- The menu also shows the GPU's total VRAM use (all programs) when `nvidia-smi` is available.
- API: `GET /api/gpu`, `POST /api/gpu/{release_wd14|load_wd14|sleep_vlm|wake_vlm|release_waifu2x}`.

## Civitai cloud training

Train on Civitai's GPUs through the [Orchestration API](https://developer.civitai.com/orchestration/) instead of your own:

1. Create a key at civitai.com → Account settings → API Keys, put it in `CIVITAI_API_KEY` in `.env` and rebuild the app container. The key stays on the server and is never sent to the browser.
2. On the project page, click "☁ Civitai training" and choose a training type. The form shows only the parameters that type supports, pre-filled with its values (what you see is exactly what is sent), and estimates the Buzz price as you type.
3. "Upload and estimate cost" uploads the images as JPEG (each one goes through Civitai's moderation; blocked images are listed) and prices the run with `whatif`. **Nothing is charged.**
4. Only "Start training" submits the run and spends Buzz. The "Training runs" list then shows progress, moderation results, a download link and sample images for every epoch, and a cancel button.

**Progress**: the home page lists running trainings from all projects (and those that finished in the last 24 hours) under "Civitai cloud training"; project cards and the Civitai button on the project page show the percentage, and clicking opens that project's "Training runs" list. It refreshes every 15 seconds:

- Queued: how many jobs are ahead and the expected start time.
- Training: percentage, current phase (loading the base model, training, uploading…), finished and remaining epochs, current and remaining steps, seconds per step, elapsed time, and the estimated time left and finish time.
- Steps and seconds per step come from Civitai's live trace (runs are submitted with `trace: "events"`). Time left ≈ remaining steps × seconds per step; sample generation and uploads between epochs can add a little. Without trace data, the finished epoch count or Civitai's own estimate is used.
- API: `GET /api/civitai/active`; MCP: `list_active_civitai_trainings`. `get_civitai_training` also includes `progress`.

**Training types**: all 26 types that Civitai's AI Toolkit can train from an image dataset.

| Group | Training types |
|---|---|
| Image | SD 1.5, SDXL, Anima, Flux.1 dev / schnell, Flux.2 Klein 4B / 9B, Chroma1-HD, ERNIE-Image, Qwen-Image (latest / 2509), Qwen Image 2.1, Z-Image Turbo / Base, Boogu, HiDream O1, Ideogram 4, Krea 2, Mage-Flow, Ming |
| Video (trainable from images) | LTX-2, LTX-2.3, LTX-2.5, Wan 2.1 / 2.2 (marked as preview by Civitai), MiniMax H3 |

- The default follows the project's base model: SD 1.5 → `sd1`; SDXL / Pony / Illustrious / NoobAI / Animagine → `sdxl`, training on SDXL 1.0, Pony Diffusion V6 XL, Illustrious-XL v0.1, NoobAI-XL eps 1.1 and Animagine XL 4.0 respectively; Anima → `anima` (Anima-Base v1.0); FLUX.1 dev → `flux1-dev`.
- Values and prices come from the [Civitai documentation](https://developer.civitai.com/orchestration/recipes/). Types the documentation does not cover (Anima, Qwen 2.1, Boogu, HiDream O1, Ideogram 4, Krea 2, Mage-Flow, Ming, LTX-2.5, MiniMax H3) use AI Toolkit's generic values; check the estimate for their price.
- Music types (ACE-Step, YuE2) need audio data and are not supported; this tool only handles images.

**Parameters** (from Civitai's `v2-consumers.json` spec; they differ per type):

| Parameter | Types |
|---|---|
| Steps, epochs, learning rate, LR scheduler, optimizer, network dim / alpha, noise offset, flip, shuffle / keep tokens, LoRA to continue from, sample prompts / negative prompt / CFG / LoRA strength | All |
| Batch size | Up to 4 for SD 1.5 / SDXL; 2 for Flux.2 Klein 4B, ERNIE, Z-Image; fixed at 1 for the rest |
| Custom base checkpoint | SD 1.5, SDXL and Anima only (Civitai fixes the base model for the other types) |
| Trigger word | SD 1.5, SDXL, Flux.1, Flux.2 Klein, Chroma, Z-Image (for other types the trigger is written at the start of every caption) |
| Min SNR γ, train text encoder, text encoder LR | SD 1.5 and SDXL only (the documentation says the other types do not train the text encoder) |

- Defaults: cosine LR scheduler; noise offset 0.1 for SDXL, 0 otherwise; Min SNR γ 5; text encoder LR 5e-5; sample LoRA strength 1.0; sample CFG uses Civitai's generation default for the base model (SDXL 7, Flux.1 3.5, Klein 5, Qwen 2.5, Anima 4…). Flux.1 schnell and MiniMax H3 have no published value and are left to Civitai.
- The base checkpoint and the LoRA to continue from accept an AIR, a model version ID, or a civitai.com URL containing `modelVersionId` (resolved with the Site API). The checkpoint's type is checked; a LoRA is checked to really be a LoRA, and for SD / Flux / Klein / Chroma / ERNIE / Qwen / Z-Image / Anima, to belong to the same base model (Wan and LTX use different names on the site and are validated by Civitai's estimate).
- Resolution, repeats, clip skip and other settings of the civitai.com trainer are not part of the API and cannot be set; Civitai picks the training resolution for the base model. Flux.2 Klein edit training (`isEditTraining`) needs paired reference images and is not supported.
- Only the images an export would include are uploaded (minor-coded + sexual images are excluded per Civitai's policy). Captions are identical to the exported `.txt` files; shuffle / keep tokens default to the tagging settings. Civitai accepts at most 1024 characters per caption: longer captions leave out their lowest-scoring WD14 tags until they fit (tags you added yourself go last), and the trigger, description, appended tags and white-block keyword are always kept.

**Uploads and retraining**

- Uploaded images are recognized by their content and remembered for about 25 days, so training again does not upload them again. Captions are sent with every training request, so **editing tags never needs a re-upload**. Images are uploaded again when they change, when "Upload long side" changes, when the API key changes, or automatically when Civitai no longer knows them.
- "Force re-upload of all images" (API / MCP: `force_upload`) re-uploads everything, for example after changing image files outside the app or to have Civitai moderate them again.
- **Retrain with these settings**: every entry in "Training runs" has this button. It fills the form with that run's training parameters, sample prompts and upload settings; your current tags are used, and the queue priority keeps the form's choice. You still estimate and confirm before anything is submitted. In the API / MCP, a run's `request` holds those parameters.

**Queue priority**: "Normal" is what the High Priority switch of the civitai.com trainer sends (the default here); "Low" is the value without that switch, which the API also uses when no priority is given; "High" is only available through the API and its effect depends on the account tier. In our tests the estimated price was the same for all three; the estimate always shows the actual price.

**Cost, cancelling and refunds**

- The estimate may include a Civitai discount; the form then also shows the undiscounted price, and the final charge may settle at that amount (charged when the run is submitted, the rest when it finishes). "Training runs" shows what was actually charged according to Civitai's transactions.
- Cancelling sends `status: canceled`, the same request the civitai.com site uses; training stops and cannot be resumed. Cancelling is asynchronous: Civitai shows "Canceled" only after the training machine stops, which can take a few minutes, and the run shows "Cancelling" until then. Civitai only documents refunds for work that has not started; after cancelling, "Training runs" shows the Buzz actually refunded (it keeps checking for an hour after the run ends).
- Prices follow Civitai's current rates (for example 0.2 Buzz per step + 10 Buzz per epoch for SDXL / SD1, and at least 80% of the default configuration's price). The estimate is the reference.

**API / MCP**: `GET /api/civitai` lists every type's `fields`, `defaults` and `max_batch`; sending a parameter the type does not support returns an error. MCP flow: `get_civitai_training_types` → `prepare_civitai_training` (takes `training_type`, `priority`, `force_upload` and the parameters above) → `get_civitai_preparation` (the estimate) → `submit_civitai_training` only after the user confirms the cost → `get_civitai_training`.

---

## Using it from other LLMs

The web UI's "API / MCP" page has ready-to-copy settings.

### MCP

Streamable HTTP endpoint: `http://<host>:7870/mcp`

```bash
# Claude Code
claude mcp add --transport http lora-tag-studio http://localhost:7870/mcp
```

```jsonc
// Cursor / VS Code / other clients that support HTTP
{ "mcpServers": { "lora-tag-studio": { "type": "http", "url": "http://localhost:7870/mcp" } } }
```

27 tools:

| Area | Tools |
|---|---|
| Base models and quick tagging | `list_profiles`, `get_tagging_guide`, `quick_tag_image` |
| Projects and import | `list_projects`, `create_project`, `get_project`, `update_project_settings`, `add_images_from_urls`, `list_server_import_folders`, `import_server_folder` |
| Tagging and editing | `start_tagging`, `get_job_status`, `list_captions`, `get_image`, `update_image_caption`, `bulk_edit_tags`, `get_tag_stats`, `detect_white_blocks` |
| Upscaling | `upscale_images`, `restore_upscaled_images` |
| Export | `export_dataset` |
| Civitai training | `get_civitai_training_types`, `prepare_civitai_training`, `get_civitai_preparation`, `submit_civitai_training`, `get_civitai_training`, `list_active_civitai_trainings` |

### REST / OpenAPI

- Swagger UI at `/docs`, spec at `/openapi.json` (works as an Open WebUI OpenAPI tool server, a GPTs action, etc.).
- A guide written for LLMs: `/llms.txt`.

```bash
# Caption a single image without creating a project
curl -X POST http://localhost:7870/api/quick-tag/json -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/a.png","profile":"pony_v6","trigger":"mychar"}'

# Full workflow
curl -X POST http://localhost:7870/api/projects -H "Content-Type: application/json" \
  -d '{"name":"my-char","profile":"illustrious","lora_type":"character","trigger":"mychar"}'
curl -X POST http://localhost:7870/api/projects/<id>/upload -F "files=@dataset.zip"
curl -X POST http://localhost:7870/api/projects/<id>/tag -H "Content-Type: application/json" -d '{"only_untagged":true}'
curl http://localhost:7870/api/jobs/<job_id>
curl -OJ "http://localhost:7870/api/projects/<id>/export?format=civitai"
```

When `API_KEY` is set, every `/api` and `/mcp` request needs `Authorization: Bearer <API_KEY>` (or `X-API-Key`). The web UI asks for the key and keeps it in a cookie.

**Response language**: API messages, base model names and guides use `?lang=`, then the `X-Lang` header, then `Accept-Language`, and fall back to `DEFAULT_LANG` (default `en`). MCP uses `DEFAULT_LANG`; `get_tagging_guide` also takes `language`. Codes: `zh-TW`, `en`, `ja`, `ko`, `zh-CN` (listed by `GET /api/i18n`).

---

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `APP_PORT` | `7870` | Published port (avoids A1111's 7860) |
| `API_KEY` | empty | Enables authentication; always set it when the server is reachable from outside |
| `PUBLIC_BASE_URL` | empty | Prefix for download links returned to LLMs, e.g. `http://192.168.1.10:7870` |
| `DEFAULT_LANG` | `en` | Language for API / MCP requests that do not specify one: `zh-TW` / `en` / `ja` / `ko` / `zh-CN` (the web UI follows the browser) |
| `WD14_MODEL` | `SmilingWolf/wd-eva02-large-tagger-v3` | Default WD14 model (can be changed per project) |
| `ORT_DEVICE` | `auto` | `auto` / `cpu` / `cuda` (also used by waifu2x) |
| `WAIFU2X_DIR` | `/models/waifu2x` | Where the waifu2x models are stored |
| `WAIFU2X_MODELS_URL` | nunif release `waifu2x_onnx_models_20250502.zip` | Zip the waifu2x models are read from (needs HTTP range support) |
| `TAG_CONCURRENCY` | `2` | Images processed at the same time (raise it for a remote VLM) |
| `HF_TOKEN` | empty | Optional Hugging Face token for faster downloads |
| `VLM_BACKEND` | `openai` | `openai` (OpenAI-compatible) / `anthropic` / `none` |
| `VLM_BASE_URL` / `VLM_MODEL` / `VLM_API_KEY` | Ollama | OpenAI-compatible endpoint |
| `VLM_TIMEOUT` | `180` | VLM request timeout in seconds |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` / `ANTHROPIC_EFFORT` | –, `claude-opus-5-5`, `low` | Used with `VLM_BACKEND=anthropic` (server-side refusal fallback enabled) |
| `ALLOW_PRIVATE_URLS` | `0` | Allow private network addresses in "Import from URL" (off to prevent SSRF) |
| `CIVITAI_API_KEY` | empty | Key for Civitai cloud training (kept on the server) |
| `JOYCAPTION_GPU_UTIL` | `0.7` | Share of VRAM vLLM reserves |
| `JOYCAPTION_VLLM_IMAGE` / `JOYCAPTION_TRANSFORMERS` | `vllm/vllm-openai:v0.30.0` / `5.16.1` | vLLM image and transformers version for JoyCaption; rebuild with `--build` after changing them |

The comments in `.env.example` are in Traditional Chinese; this table covers the same variables.

Data lives in `./data` (SQLite database, images, thumbnails, export zips) and `./import` (read-only folder for server-side imports). Both are excluded from git.

---

## Project structure

```
app/
  main.py            FastAPI entry point, API key check, language negotiation, MCP mount, /llms.txt
  i18n.py            Locale loading, language negotiation, translation
  locales/*.json     Text for each language (server: backend / guides, ui: web UI)  ← translations live here
  api.py             REST API
  mcp_server.py      MCP tools
  profiles.py        Base model defaults, rating mappings, tagging guides  ← add or adjust base models here
  pipeline.py        Tagging pipeline for one image
  jobs.py            Background job queue (tagging, upscaling)
  services.py        Logic shared by the API and MCP
  exporter.py        Civitai / kohya / jsonl export
  civitai.py         Civitai cloud training (Orchestration API)
  upscale.py         waifu2x upscaling: model download, tiled inference, backup / restore
  storage.py         Uploads, folder / zip import, thumbnails
  db.py              SQLite
  tagging/
    wd14.py          WD14 ONNX inference
    vlm.py           OpenAI-compatible / Claude descriptions
    postprocess.py   Thresholds, ordering, pruning groups, blacklist, caption assembly
    blocks.py        Detection of white masking blocks
web/                 Web UI (plain HTML / CSS / JS, no build step)
docker/
  joycaption.Dockerfile  vLLM image for JoyCaption
scripts/
  check_locales.py   Checks that every locale has the same keys and {placeholders} as en.json
```

## Development

Run without Docker (Python 3.12):

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt onnxruntime
DATA_DIR=./data IMPORT_DIR=./import VLM_BACKEND=none uvicorn app.main:app --reload --port 7870
```

**Translations**: edit `app/locales/<code>.json` (missing keys fall back to English). To add a language, add a new JSON file; `_meta.order` sets its position in the menu. Then run `python3 scripts/check_locales.py` (standard library only) and rebuild with `--build`.

---

## Troubleshooting

- **VLM shows as unreachable**: make sure you started with `--profile vlm` and that `docker compose logs ollama-pull` shows the model has finished downloading.
- **VLM returns 404**: the model name does not match `ollama list`.
- **The GPU is not used**: if the top-right corner shows "WD14 · CPU", check that you used `docker-compose.gpu.yml` and rebuilt with `--build`, and that `docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi` works.
- **JoyCaption runs out of memory**: lower `JOYCAPTION_GPU_UTIL` or use a smaller VLM; the GPU build of WD14 uses about 1–2 GB. To share the GPU with other tools, put JoyCaption to sleep from the "VRAM ▾" menu when you are not tagging.
- **The VRAM menu says the VLM cannot sleep**: the joycaption container was created without sleep mode; recreate it with `docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile joycaption up -d joycaption`.
- **`docker compose down` shows `Network lora-tag-studio_default  Resource is still in use`**: a container from a profile (JoyCaption or Ollama) is still running. Run `docker compose --profile vlm --profile joycaption down` (see [Stopping](#stopping)).
- **JoyCaption fails with `cannot import name 'PixtralRotaryEmbedding'`**: an unpatched `vllm/vllm-openai` image is in use. Check that the joycaption service in `docker-compose.yml` uses `build:` and run `docker compose ... --profile joycaption up -d --build`.
- **The web UI shows key names such as `common.server_unreachable`**: the image has no locale files (`app/locales/*.json`); rebuild with `--build`. `docker compose logs app` lists the loaded languages at startup.
- **Upscaling fails at "Preparing the waifu2x models"**: the app has to reach github.com once to download the models; `docker compose logs app` shows the error.
- **Captions longer than 75 tokens**: CLIP in SD 1.5 / SDXL reads 75 tokens at a time; lower "Max tags" or raise the threshold. kohya can use `--max_token_length=225`.

## Notes

- This project is not affiliated with Civitai. Cloud training spends Buzz from your own Civitai account; check the estimate before you confirm.
- You are responsible for having the rights to the images you train on and for following the content rules of the services you use.
- Models (WD14, waifu2x, JoyCaption, Ollama models) are downloaded from their publishers at runtime and are subject to their own licenses.

## License

[MIT](LICENSE)
