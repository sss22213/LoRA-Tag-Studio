"""REST API（/api）。所有端點都有 operation_id 與說明，可直接當作 LLM 的 OpenAPI 工具。"""
from __future__ import annotations

import base64
import binascii
import zipfile
from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import civitai, exporter, i18n, jobs, services, storage
from .config import settings
from .i18n import t, tr
from .pipeline import quick_tag
from .profiles import (
    BASE_SETTINGS,
    PROFILES,
    caption_modes_view,
    guide_markdown,
    list_profiles,
    lora_types_view,
    profile_view,
)
from .tagging import vlm
from .tagging.postprocess import PRUNE_GROUPS
from .tagging.wd14 import loaded_models, wd14_models_view

router = APIRouter(prefix="/api")

ProfileKey = Literal[tuple(PROFILES)]  # type: ignore[valid-type]
LoraType = Literal["character", "style", "concept"]
CaptionMode = Literal["tags", "natural", "hybrid", "trigger_only"]
VlmTagMode = Literal["off", "extra", "merge", "only"]


# ------------------------------------------------------------------ models
class ProjectCreate(BaseModel):
    name: str = Field(..., description="Project / dataset name")
    profile: ProfileKey = Field("illustrious", description="Target base model profile (see GET /api/profiles)")
    lora_type: LoraType = Field("character", description="What the LoRA should learn")
    trigger: str = Field("", description="Trigger word placed at the start of every caption")
    class_word: str = Field("", description="Optional class word, e.g. 1girl / woman / style")
    settings: dict[str, Any] | None = Field(None, description="Optional overrides of project settings")


class ProjectUpdate(BaseModel):
    name: str | None = None
    settings: dict[str, Any] | None = Field(None, description="Partial settings to merge (see GET /api/system → settings_schema)")
    reset_to_profile: bool = Field(False, description="If settings.profile changes, reset other settings to that profile's defaults")


class ImageUpdate(BaseModel):
    tags: list[str] | str | None = Field(None, description="Full ordered tag list (or comma separated string)")
    nl_caption: str | None = Field(None, description="Natural-language caption")
    has_blocks: bool | None = Field(None, description="Manually mark whether the image has white blocks (overrides "
                                                      "detection)")
    clear_blocks_override: bool = Field(False, description="Go back to the detection result")


class BlockScanRequest(BaseModel):
    force: bool = Field(False, description="Re-detect all images (default: only images not scanned yet)")


class IdList(BaseModel):
    ids: list[str]


class UpscaleRequest(BaseModel):
    ids: list[str] | None = Field(None, description="Images to upscale (default: every image whose short side is "
                                                     "below min_side and that was not upscaled yet)")
    min_side: int = Field(1024, ge=64, le=8192, description="Target short side in px: picks images when ids is "
                                                            "omitted (except scale=1), and scale=auto uses 4x only if "
                                                            "2x stays below it")
    style: Literal["art", "art_scan", "photo"] = Field("art", description="art = illustrations / anime, art_scan = "
                                                                          "scans (halftone, paper), photo = photos")
    noise: str | int = Field("auto", description="JPEG noise reduction: auto (level 1 for JPEG / lossy WebP, none "
                                                 "otherwise) | none | 0 | 1 | 2 | 3")
    scale: str | int = Field("auto", description="auto (2x, 4x when 2x stays below min_side) | 2 | 4 | 1 = noise "
                                                 "reduction only, size unchanged (without ids: JPEG / lossy WebP "
                                                 "images of any size that were not processed yet)")


class UpscaleRestoreRequest(BaseModel):
    ids: list[str] | None = Field(None, description="Images to restore (default: all upscaled images)")


class UrlImport(BaseModel):
    urls: list[str] = Field(..., description="http(s) image URLs to download into the project")


class ServerImport(BaseModel):
    path: str = Field("", description="Sub folder inside the server import directory (see GET /api/import-dirs)")
    recursive: bool = True


class TagRequest(BaseModel):
    ids: list[str] | None = Field(None, description="Image ids to tag; omit for all images")
    only_untagged: bool = Field(False, description="Only tag images that are pending or failed")


class BulkRequest(BaseModel):
    action: Literal["add", "remove", "replace", "reapply", "filter", "clear_nl"]
    ids: list[str] | None = Field(None, description="Image ids; omit for all images in the project")
    tags: list[str] | str | None = Field(None, description="Tags for add/remove (remove supports * wildcards)")
    find: str = Field("", description="Tag to find (replace action)")
    replace: str = Field("", description="Replacement tag(s), comma separated (replace action)")
    position: Literal["front", "back"] = "back"


class CivitaiTrainRequest(BaseModel):
    """Fields differ per training type: GET /api/civitai lists each type's `fields` and `defaults`.
    Omitted (null) numbers use Civitai's defaults; type-specific fields sent to a type without them are rejected."""
    training_type: str | None = Field(None, description="Civitai training type id from GET /api/civitai (e.g. sdxl, anima, "
                                                        "flux1-dev, flux2klein-4b, qwen, wan-2.2). Empty = default for "
                                                        "the project's base model")
    base_model: str = Field("", description="sd1 / sdxl / anima only. Base checkpoint to train on: an AIR, a Civitai model "
                                            "version ID or a civitai.com URL with modelVersionId. Empty = default for "
                                            "the project's base model")
    steps: int | None = Field(None, ge=1, le=10000, description="Total training steps (main driver of the Buzz price)")
    epochs: int | None = Field(None, ge=1, le=20, description="Saved checkpoints (each adds a per-epoch fee)")
    batch_size: int | None = Field(None, ge=1, le=4, description="Clamped to the type's max_batch (1, 2 or 4)")
    lr: float | None = Field(None, gt=0, le=1)
    lr_scheduler: Literal[civitai.LR_SCHEDULERS] | None = None  # type: ignore[valid-type]
    optimizer: Literal[civitai.OPTIMIZERS] | None = None  # type: ignore[valid-type]
    network_dim: int | None = Field(None, ge=1, le=256)
    network_alpha: int | None = Field(None, ge=1, le=256)
    noise_offset: float | None = Field(None, ge=0, le=1)
    flip_augmentation: bool | None = Field(None, description="Random horizontal flips")
    shuffle_tokens: bool | None = Field(None, description="Shuffle caption tags (default: from the caption structure)")
    keep_tokens: int | None = Field(None, ge=0, le=10, description="Tags kept at the front when shuffling")
    trigger_word: str | None = Field(None, description="Types with trigger_word only (sd1, sdxl, flux1, flux2klein, "
                                                       "chroma, zimage*). Default: the project's trigger")
    min_snr_gamma: int | None = Field(None, ge=0, le=20, description="sd1 / sdxl only")
    train_text_encoder: bool | None = Field(None, description="sd1 / sdxl only")
    text_encoder_lr: float | None = Field(None, gt=0, le=1, description="sd1 / sdxl only; used when train_text_encoder")
    continue_from: str = Field("", description="Continue training a LoRA of the same type: AIR, model version ID or "
                                               "civitai.com URL. Empty = start from the base model")
    priority: Literal[civitai.PRIORITIES] = Field(civitai.DEFAULT_PRIORITY, description=(  # type: ignore[valid-type]
        "Queue priority. normal = the civitai.com trainer's High Priority switch; low = switch off (Civitai's default "
        "when omitted); high = API only, effect depends on the account tier. The estimate shows the price"))
    force_upload: bool = Field(False, description="Re-upload every image even if it was uploaded before (each one goes "
                                                  "through Civitai moderation again). Not needed after editing tags: "
                                                  "captions are sent with every training request")
    max_side: int = Field(2048, ge=512, le=4096, description="Images are uploaded as JPEG, downscaled to this long side")
    only_done: bool = Field(True, description="Only upload images whose tagging is finished")
    allow_mature: bool | None = Field(None, description="Allow mature content (default: on when the dataset has NSFW ratings)")
    sample_prompts: list[str] | None = Field(None, max_length=5, description="Preview prompts (default: dataset captions)")
    sample_negative: str | None = Field(None, description="Negative prompt for previews (default: the base model's; "
                                                          "empty string = none)")
    sample_cfg: float | None = Field(None, ge=0, le=30, description="CFG scale for previews (default: Civitai's)")
    sample_strength: float | None = Field(None, ge=0, le=2, description="LoRA strength in previews (default 1.0)")


class ExportRequest(BaseModel):
    format: Literal["civitai", "kohya", "jsonl"] = "civitai"
    image_format: Literal["original", "png", "jpg", "webp"] = "original"
    max_side: int = Field(0, ge=0, description="Downscale so the longest side <= this (0 = keep)")
    repeats: int = Field(10, ge=1, description="kohya folder repeats")
    naming: Literal["original", "sequential"] = "original"
    only_done: bool = False


class QuickTagJSON(BaseModel):
    image_url: str | None = Field(None, description="http(s) URL of the image")
    image_base64: str | None = Field(None, description="Base64 encoded image (data URL prefix allowed)")
    profile: ProfileKey = "illustrious"
    lora_type: LoraType = "character"
    trigger: str = ""
    caption_mode: CaptionMode | None = None
    use_vlm: bool | None = None
    vlm_nsfw: bool | None = None
    vlm_tags: VlmTagMode | None = Field(
        None, description="Danbooru tags from the VLM (JoyCaption): off | extra (add character/copyright/artist) | "
                          "merge (WD14 + all VLM tags) | only (VLM tags only)")
    general_threshold: float | None = Field(None, ge=0, le=1)
    character_threshold: float | None = Field(None, ge=0, le=1)
    max_tags: int | None = Field(None, ge=0)
    blacklist: str | None = None


# ------------------------------------------------------------------ system
@router.get("/health", operation_id="health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@router.get("/i18n", operation_id="list_languages", summary="Available UI / content languages")
def languages() -> dict[str, Any]:
    return {"default": i18n.default_lang(), "current": i18n.get_lang(), "languages": list(i18n.available().values())}


@router.get("/i18n/{lang}", operation_id="get_ui_translations", include_in_schema=False)
def ui_translations(lang: str) -> dict[str, Any]:
    code = i18n.normalize(lang)
    if code is None:
        raise HTTPException(404, t("msg.unknown_language", code=lang))
    return i18n.ui_catalog(code)


@router.get("/system", operation_id="get_system_info", summary="Tagger / VLM status and all option lists")
def system_info() -> dict[str, Any]:
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except Exception:  # noqa: BLE001
        providers = []
    return {
        "wd14": {"default_model": settings.wd14_default_model, "models": wd14_models_view(), "loaded": loaded_models(),
                 "available_providers": providers, "device": settings.ort_device},
        "vlm": vlm.status(),
        "civitai": civitai.info(),
        "lora_types": lora_types_view(),
        "caption_modes": caption_modes_view(),
        "prune_groups": {k: tr(f"prune_groups.{k}", default=k) for k in PRUNE_GROUPS},
        "export_formats": exporter.export_formats_view(),
        "bulk_actions": services.bulk_actions_view(),
        "upscale": services.upscale_options(),
        "settings_schema": BASE_SETTINGS,
        "nsfw_common": tr("nsfw_common", default=""),
        "lang": i18n.get_lang(),
        "auth": bool(settings.api_key),
    }


class GpuActionRequest(BaseModel):
    model: str | None = Field(None, description="load_wd14 only: WD14 model repo (default: the server default)")


@router.get("/gpu", operation_id="get_gpu_status",
            summary="VRAM: loaded WD14 / waifu2x models, whether the VLM (vLLM) is sleeping, GPU memory use")
def gpu_status() -> dict[str, Any]:
    return services.gpu_status()


@router.post("/gpu/{action}", operation_id="manage_vram",
             summary="Free or load VRAM: release_wd14 | load_wd14 | sleep_vlm | wake_vlm | release_waifu2x",
             description="release_wd14 unloads the WD14 models (reloaded automatically on the next tagging). "
                         "release_waifu2x unloads the upscaling models (also done automatically after each "
                         "upscaling job). "
                         "sleep_vlm puts vLLM (JoyCaption) to sleep: weights move to system RAM and the VRAM is "
                         "freed; it wakes automatically before the next VLM call. Releasing is refused while a "
                         "tagging or upscaling job is queued or running. vLLM must run with --enable-sleep-mode and "
                         "VLLM_SERVER_DEV_MODE=1 (the bundled joycaption service does).")
def gpu_action(action: str, body: GpuActionRequest | None = None) -> dict[str, Any]:
    return services.gpu_action(action, body.model if body else None)


@router.get("/profiles", operation_id="list_profiles", summary="List base-model tagging profiles")
def get_profiles() -> list[dict[str, Any]]:
    return list_profiles()


@router.get("/profiles/{key}", operation_id="get_profile", summary="Profile details: guide, defaults, rating map")
def get_profile_detail(key: str) -> dict[str, Any]:
    if key not in PROFILES:
        raise HTTPException(404, t("msg.unknown_profile", profile=key, available=", ".join(PROFILES)))
    return profile_view(key)


@router.get("/profiles/{key}/guide.md", operation_id="get_profile_guide_markdown", response_class=PlainTextResponse,
            summary="Tagging guide for a base model as Markdown")
def get_profile_guide(key: str) -> str:
    if key not in PROFILES:
        raise HTTPException(404, t("msg.unknown_profile", profile=key, available=", ".join(PROFILES)))
    return guide_markdown(key)


# ------------------------------------------------------------------ projects
@router.get("/projects", operation_id="list_projects")
def list_projects() -> list[dict[str, Any]]:
    from . import db

    return [services.project_out(p) for p in db.list_projects()]


@router.post("/projects", operation_id="create_project", status_code=201)
def create_project(body: ProjectCreate) -> dict[str, Any]:
    return services.create_project(body.name, body.profile, body.lora_type, body.trigger, body.class_word,
                                   **(body.settings or {}))


@router.get("/projects/{pid}", operation_id="get_project")
def get_project(pid: str) -> dict[str, Any]:
    return services.project_out(services.require_project(pid))


@router.patch("/projects/{pid}", operation_id="update_project")
def update_project(pid: str, body: ProjectUpdate) -> dict[str, Any]:
    return services.update_project(pid, body.name, body.settings, body.reset_to_profile)


@router.delete("/projects/{pid}", operation_id="delete_project")
def delete_project(pid: str) -> dict[str, Any]:
    services.delete_project(pid)
    return {"deleted": pid}


# ------------------------------------------------------------------ images
@router.get("/projects/{pid}/images", operation_id="list_images",
            summary="List images with tags, natural caption and the final training caption")
def list_images(pid: str, status: str | None = Query(None, description="pending/queued/processing/done/error"),
                search: str | None = Query(None, description="Comma separated tags / text; * wildcard")) -> list[dict[str, Any]]:
    return services.list_images(pid, status, search)


@router.post("/projects/{pid}/upload", operation_id="upload_images",
             summary="Upload images / folders / zip files. Same-name .txt files are imported as existing captions")
async def upload(pid: str, files: list[UploadFile] = File(...),
                 paths: list[str] | None = Form(None, description="Relative paths (same order as files)")) -> dict[str, Any]:
    project = services.require_project(pid)
    trigger = project["settings"].get("trigger", "")
    entries: list[storage.Entry] = []
    zips: list[UploadFile] = []
    for i, f in enumerate(files):
        name = (paths[i] if paths and i < len(paths) and paths[i] else f.filename) or f"file_{i}"
        if name.lower().endswith(".zip"):
            zips.append(f)  # zip 直接從暫存檔讀取，不整包載入記憶體
        else:
            entries.append(storage.bytes_entry(name, await f.read()))

    def work() -> dict[str, Any]:
        result = storage.import_entries(pid, entries, trigger=trigger)
        for z in zips:
            try:
                r = storage.import_zip(pid, z.file, trigger=trigger)
            except zipfile.BadZipFile:
                result["skipped"].append({"file": z.filename, "reason": t("msg.zip_corrupt")})
                continue
            result["added"] += r["added"]
            result["added_ids"] += r["added_ids"]
            result["skipped"] += r["skipped"]
        return result

    return await run_in_threadpool(work)


@router.post("/projects/{pid}/import-urls", operation_id="import_image_urls", summary="Download images from URLs")
def import_urls(pid: str, body: UrlImport) -> dict[str, Any]:
    return services.import_urls(pid, body.urls)


@router.get("/import-dirs", operation_id="list_server_import_dirs",
            summary="Folders available in the server import directory (mounted ./import)")
def import_dirs() -> list[dict[str, Any]]:
    return storage.list_import_dirs()


@router.post("/projects/{pid}/import-server", operation_id="import_server_folder",
             summary="Import a folder from the server import directory")
def import_server(pid: str, body: ServerImport) -> dict[str, Any]:
    project = services.require_project(pid)
    try:
        entries = storage.server_dir_entries(body.path, body.recursive)
        return storage.import_entries(pid, entries, trigger=project["settings"].get("trigger", ""))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/images/{iid}/file", operation_id="get_image_file", include_in_schema=False)
def image_file(iid: str) -> FileResponse:
    img = services.require_image(iid)
    return FileResponse(storage.image_path(img), headers={"Cache-Control": "private, max-age=86400"})


@router.get("/images/{iid}/thumb", operation_id="get_image_thumb", include_in_schema=False)
def image_thumb(iid: str) -> FileResponse:
    img = services.require_image(iid)
    path = storage.thumb_path(img)
    return FileResponse(path if path.exists() else storage.image_path(img),
                        headers={"Cache-Control": "private, max-age=604800"})


@router.patch("/projects/{pid}/images/{iid}", operation_id="update_image_caption",
              summary="Replace an image's tags and/or natural-language caption")
def update_image(pid: str, iid: str, body: ImageUpdate) -> dict[str, Any]:
    return services.update_image(pid, iid, body.tags, body.nl_caption, body.has_blocks, body.clear_blocks_override)


@router.post("/projects/{pid}/blocks/scan", operation_id="detect_white_blocks",
             summary="Detect white blocks (rectangles used to cover other people) in the project's images. Images with "
                     "blocks get the project's block_tag at the end of the caption when block_tag_auto is on")
def scan_blocks(pid: str, body: BlockScanRequest) -> dict[str, Any]:
    return services.scan_blocks(pid, body.force)


@router.post("/projects/{pid}/images/delete", operation_id="delete_images")
def delete_images(pid: str, body: IdList) -> dict[str, Any]:
    return {"deleted": services.delete_images(pid, body.ids)}


# ------------------------------------------------------------------ tagging
@router.post("/projects/{pid}/tag", operation_id="start_tagging",
             summary="Start a background tagging job (WD14 + optional VLM). Poll GET /api/jobs/{job_id}")
def tag(pid: str, body: TagRequest | None = None) -> dict[str, Any]:
    body = body or TagRequest()
    return services.start_tagging(pid, body.ids, body.only_untagged)


@router.post("/projects/{pid}/upscale", operation_id="upscale_images",
             summary="Upscale low-resolution images, or only remove JPEG noise (scale=1), with waifu2x "
                     "(background job, poll GET /api/jobs/{id})",
             description="Uses nunif's swin_unet waifu2x ONNX models (downloaded on first use, ~17–19 MB each) on "
                         "the same GPU / CPU as WD14. Results are saved as PNG; the original is backed up and can be "
                         "restored with POST /upscale/restore. Re-upscaling always starts from the original. Tags "
                         "do not need to be redone; white blocks are re-detected (manual marks are kept).")
def upscale(pid: str, body: UpscaleRequest | None = None) -> dict[str, Any]:
    b = body or UpscaleRequest()
    return services.start_upscale(pid, ids=b.ids, min_side=b.min_side, style=b.style, noise=b.noise, scale=b.scale)


@router.post("/projects/{pid}/upscale/restore", operation_id="restore_upscaled_images",
             summary="Put the original (pre-upscale) images back")
def upscale_restore(pid: str, body: UpscaleRestoreRequest | None = None) -> dict[str, Any]:
    return services.restore_upscaled(pid, ids=body.ids if body else None)


@router.get("/jobs/{jid}", operation_id="get_job")
def get_job(jid: str) -> dict[str, Any]:
    job = jobs.get(jid)
    if job is None:
        raise HTTPException(404, t("msg.job_not_found"))
    return job.to_dict()


@router.post("/jobs/{jid}/cancel", operation_id="cancel_job")
def cancel_job(jid: str) -> dict[str, Any]:
    job = jobs.cancel(jid)
    if job is None:
        raise HTTPException(404, t("msg.job_not_found"))
    return job.to_dict()


@router.get("/projects/{pid}/jobs", operation_id="list_project_jobs")
def project_jobs(pid: str) -> list[dict[str, Any]]:
    services.require_project(pid)
    return [j.to_dict() for j in jobs.list_for_project(pid)[:20]]


@router.get("/projects/{pid}/stats", operation_id="get_tag_stats", summary="Tag frequency across the dataset")
def stats(pid: str) -> dict[str, Any]:
    return services.tag_stats(pid)


@router.post("/projects/{pid}/bulk", operation_id="bulk_edit_tags", summary="Bulk add/remove/replace/re-apply tags")
def bulk(pid: str, body: BulkRequest) -> dict[str, Any]:
    return services.bulk_edit(pid, body.action, body.ids, body.tags, body.find, body.replace, body.position)


@router.get("/projects/{pid}/captions", operation_id="list_captions",
            summary="Final training captions exactly as they will be exported")
def captions(pid: str, limit: int = Query(1000, ge=1, le=5000), offset: int = Query(0, ge=0)) -> dict[str, Any]:
    return services.captions(pid, limit, offset)


# ------------------------------------------------------------------ export
def _export(pid: str, body: ExportRequest):
    services.require_project(pid)
    try:
        return exporter.export_dataset(pid, body.format, body.image_format, body.max_side, body.repeats,
                                       body.naming, body.only_done)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/projects/{pid}/export", operation_id="download_dataset_zip", response_class=FileResponse,
            summary="Build and download the training zip (Civitai: images + same-name .txt)")
def export_get(pid: str, format: Literal["civitai", "kohya", "jsonl"] = "civitai",
               image_format: Literal["original", "png", "jpg", "webp"] = "original", max_side: int = 0,
               repeats: int = 10, naming: Literal["original", "sequential"] = "original",
               only_done: bool = False) -> FileResponse:
    path, excluded = _export(pid, ExportRequest(format=format, image_format=image_format, max_side=max_side,
                                                repeats=repeats, naming=naming, only_done=only_done))
    return FileResponse(path, filename=path.name, media_type="application/zip",
                        headers={"X-Excluded-Count": str(len(excluded))})


@router.post("/projects/{pid}/export", operation_id="export_dataset",
             summary="Build the training zip and return a download URL (for agents)")
def export_post(pid: str, body: ExportRequest) -> dict[str, Any]:
    path, excluded = _export(pid, body)
    return {"download_url": services.absolute_url(f"/api/exports/{path.name}"), "file": path.name,
            "size_bytes": path.stat().st_size, "excluded": excluded}


@router.get("/exports/{name}", operation_id="get_export_file", include_in_schema=False)
def export_file(name: str) -> FileResponse:
    path = (settings.exports_dir / name).resolve()
    if path.parent != settings.exports_dir.resolve() or not path.is_file():
        raise HTTPException(404, t("msg.export_not_found"))
    return FileResponse(path, filename=path.name, media_type="application/zip")


# ------------------------------------------------------------------ quick tag
def _quick(im, **kw: Any) -> dict[str, Any]:
    overrides = {k: v for k, v in kw.items() if v is not None and k not in ("profile", "lora_type")}
    result = quick_tag(im, kw.get("profile") or "illustrious", kw.get("lora_type") or "character", **overrides)
    result.pop("settings", None)
    return result


@router.post("/quick-tag", operation_id="quick_tag_upload",
             summary="Tag a single uploaded image without creating a project")
async def quick_tag_upload(file: UploadFile = File(...), profile: str = Form("illustrious"),
                           lora_type: str = Form("character"), trigger: str = Form(""),
                           caption_mode: str | None = Form(None), use_vlm: bool | None = Form(None),
                           vlm_nsfw: bool | None = Form(None), vlm_tags: VlmTagMode | None = Form(None)) -> dict[str, Any]:
    if profile not in PROFILES:
        raise HTTPException(400, t("msg.unknown_profile", profile=profile, available=", ".join(PROFILES)))
    im = services.decode_image(await file.read())
    return await run_in_threadpool(_quick, im, profile=profile, lora_type=lora_type, trigger=trigger,
                                   caption_mode=caption_mode, use_vlm=use_vlm, vlm_nsfw=vlm_nsfw, vlm_tags=vlm_tags)


@router.post("/quick-tag/json", operation_id="quick_tag",
             summary="Tag a single image given by URL or base64, returns the training caption for the chosen base model")
def quick_tag_json(body: QuickTagJSON) -> dict[str, Any]:
    if body.image_base64:
        data = body.image_base64.split(",", 1)[1] if body.image_base64.startswith("data:") else body.image_base64
        try:
            im = services.decode_image(base64.b64decode(data, validate=False))
        except binascii.Error as e:
            raise HTTPException(400, t("msg.invalid_base64")) from e
    elif body.image_url:
        im = services.fetch_image(body.image_url)
    else:
        raise HTTPException(400, t("msg.image_required"))
    return _quick(im, **body.model_dump(exclude={"image_url", "image_base64"}))


# ------------------------------------------------------------------ Civitai 雲端訓練
@router.get("/civitai", operation_id="get_civitai_training_info",
            summary="Whether Civitai cloud training is configured, and the training ecosystem per base model")
def civitai_info() -> dict[str, Any]:
    return civitai.info()


@router.post("/projects/{pid}/civitai/prepare", operation_id="prepare_civitai_training",
             summary="Upload the dataset to Civitai and estimate the Buzz cost (does not start training)")
def civitai_prepare(pid: str, body: CivitaiTrainRequest) -> dict[str, Any]:
    services.require_project(pid)
    try:
        return civitai.prepare(pid, body.model_dump()).to_dict()
    except ValueError as e:  # 沒有可上傳的圖片
        raise HTTPException(400, str(e)) from e


@router.get("/civitai/prepare/{prep_id}", operation_id="get_civitai_preparation",
            summary="Upload progress and cost estimate of a Civitai training preparation")
def civitai_prep_status(prep_id: str) -> dict[str, Any]:
    return civitai.get_prep(prep_id).to_dict()


@router.post("/civitai/prepare/{prep_id}/submit", operation_id="submit_civitai_training",
             summary="Start the prepared Civitai training run (spends Buzz)")
def civitai_submit(prep_id: str) -> dict[str, Any]:
    return civitai.submit(prep_id)


@router.get("/projects/{pid}/civitai/runs", operation_id="list_civitai_training_runs",
            summary="Civitai training runs of a project with status, epochs and LoRA download links")
def civitai_runs(pid: str) -> list[dict[str, Any]]:
    services.require_project(pid)
    return civitai.list_runs(pid)


@router.get("/civitai/active", operation_id="list_active_civitai_trainings",
            summary="Civitai training runs in progress in any project (plus runs finished in the last 24 h), "
                    "with progress: percent, epochs / steps done and estimated remaining seconds")
def civitai_active() -> list[dict[str, Any]]:
    return civitai.active_runs()


@router.get("/civitai/runs/{workflow_id}", operation_id="get_civitai_training_run")
def civitai_run(workflow_id: str) -> dict[str, Any]:
    return civitai.refresh_run(workflow_id)


@router.post("/civitai/runs/{workflow_id}/cancel", operation_id="cancel_civitai_training_run")
def civitai_cancel(workflow_id: str) -> dict[str, Any]:
    return civitai.cancel_workflow(workflow_id)
