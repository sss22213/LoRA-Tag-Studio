"""MCP (Model Context Protocol) 伺服器，掛載於 /mcp（Streamable HTTP）。

讓 Claude Desktop / Claude Code / Cursor / Open WebUI 等 LLM 客戶端直接操作標註流程。
"""
from __future__ import annotations

import base64
import binascii
import functools
from typing import Any, Callable

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import civitai, db, exporter, jobs, services, storage
from .pipeline import quick_tag
from .i18n import use_lang
from .profiles import PROFILES, guide_markdown, list_profiles as _list_profiles
from .tagging.vlm import VLMError

INSTRUCTIONS = """\
LoRA Tag Studio prepares image datasets for LoRA training (Civitai / kohya).
Typical workflow:
1. list_profiles → pick the base model the LoRA targets (pony_v6, illustrious, noobai_xl, anima, flux1_dev, ...).
2. get_tagging_guide(profile) to learn the caption conventions for that model.
3. create_project(name, profile, lora_type, trigger).
4. add_images_from_urls or import_server_folder to add images. Optionally upscale_images for low-resolution
   images (short side below ~1024 px) → poll get_job_status.
5. start_tagging → poll get_job_status until status is done.
6. list_captions / get_tag_stats to review; fix with bulk_edit_tags or update_image_caption.
7. export_dataset → returns a zip download URL ready for the Civitai trainer.
8. Optional cloud training on Civitai (needs CIVITAI_API_KEY on the server, spends the user's Buzz):
   get_civitai_training_types (parameter fields differ per type) → prepare_civitai_training → poll
   get_civitai_preparation until status is "ready" → tell the user the estimated
   Buzz cost and ask for confirmation → only then submit_civitai_training → poll get_civitai_training.
For a single image without a project use quick_tag_image.
"""

mcp = MCPServer(name="lora-tag-studio", title="LoRA Tag Studio", instructions=INSTRUCTIONS)


def _tool(fn: Callable[..., Any]) -> Callable[..., Any]:
    """註冊 MCP 工具；可預期的錯誤轉成 ToolError，讓 LLM 看得到原因。"""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except (services.NotFound, services.BadRequest, civitai.CivitaiError, ValueError, KeyError, VLMError,
                binascii.Error) as e:
            raise ToolError(str(e)) from e

    return mcp.tool()(wrapper)


@_tool
def list_profiles() -> list[dict[str, Any]]:
    """List supported base-model profiles (SD1.5, SDXL, Pony, Illustrious, NoobAI, Animagine, Anima, Flux) and their caption style."""
    return _list_profiles()


@_tool
def get_tagging_guide(profile: str, language: str = "") -> str:
    """Markdown guide for a base model: caption style, tag order, rating/NSFW tags, Civitai training settings, A1111 prompt template.

    language: en | zh-TW | zh-CN | ja | ko (empty = server default).
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile}; available: {', '.join(PROFILES)}")
    with use_lang(language or None):
        return guide_markdown(profile)


@_tool
def quick_tag_image(
    image_url: str = "",
    image_base64: str = "",
    profile: str = "illustrious",
    lora_type: str = "character",
    trigger: str = "",
    caption_mode: str = "",
    use_vlm: bool | None = None,
    nsfw_captions: bool | None = None,
    vlm_tags: str = "",
) -> dict[str, Any]:
    """Tag one image (by http URL or base64) and return the training caption for the given base model profile.

    caption_mode: tags | natural | hybrid | trigger_only (empty = profile default).
    nsfw_captions: let the VLM describe adult content explicitly (needs an uncensored VLM).
    vlm_tags: Danbooru tags from the VLM (JoyCaption): off | extra (add character/copyright/artist tags to WD14's)
              | merge (WD14 + all VLM tags) | only (VLM tags only). Empty = off.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile}")
    if image_base64:
        data = image_base64.split(",", 1)[1] if image_base64.startswith("data:") else image_base64
        im = services.decode_image(base64.b64decode(data))
    elif image_url:
        im = services.fetch_image(image_url)
    else:
        raise ValueError("image_url or image_base64 is required")
    overrides: dict[str, Any] = {"trigger": trigger}
    if caption_mode:
        overrides["caption_mode"] = caption_mode
    if use_vlm is not None:
        overrides["use_vlm"] = use_vlm
    if nsfw_captions is not None:
        overrides["vlm_nsfw"] = nsfw_captions
    if vlm_tags:
        overrides["vlm_tags"] = vlm_tags
    result = quick_tag(im, profile, lora_type, **overrides)
    result.pop("settings", None)
    return result


@_tool
def list_projects() -> list[dict[str, Any]]:
    """List dataset projects with image counts."""
    return [
        {"id": p["id"], "name": p["name"], "profile": p["settings"].get("profile"),
         "trigger": p["settings"].get("trigger"), "images": p["image_count"], "tagged": p["done_count"]}
        for p in db.list_projects()
    ]


@_tool
def create_project(name: str, profile: str = "illustrious", lora_type: str = "character", trigger: str = "",
                   class_word: str = "") -> dict[str, Any]:
    """Create a dataset project. lora_type: character | style | concept. Settings default to the profile's recommendations."""
    return services.create_project(name, profile, lora_type, trigger, class_word)


@_tool
def update_project_settings(project_id: str, settings: dict[str, Any], reset_to_profile: bool = False) -> dict[str, Any]:
    """Merge settings into a project (e.g. trigger, caption_mode, general_threshold, prune_groups, blacklist,
    include_rating, prefix_tags, use_vlm, vlm_nsfw). Use get_project to see all keys."""
    return services.update_project(project_id, None, settings, reset_to_profile)


@_tool
def get_project(project_id: str) -> dict[str, Any]:
    """Project details including all current settings and any running job."""
    return services.project_out(services.require_project(project_id))


@_tool
def add_images_from_urls(project_id: str, urls: list[str]) -> dict[str, Any]:
    """Download images (http/https) into a project."""
    return services.import_urls(project_id, urls)


@_tool
def list_server_import_folders() -> list[dict[str, Any]]:
    """Folders available in the server-side import directory."""
    return storage.list_import_dirs()


@_tool
def import_server_folder(project_id: str, path: str = "", recursive: bool = True) -> dict[str, Any]:
    """Import all images (and same-name .txt captions) from a server import folder into a project."""
    p = services.require_project(project_id)
    return storage.import_entries(project_id, storage.server_dir_entries(path, recursive),
                                  trigger=p["settings"].get("trigger", ""))


@_tool
def start_tagging(project_id: str, only_untagged: bool = True, image_ids: list[str] | None = None) -> dict[str, Any]:
    """Start background tagging (WD14 tags + optional VLM caption). Returns a job; poll get_job_status."""
    return services.start_tagging(project_id, image_ids, only_untagged)


@_tool
def get_job_status(job_id: str) -> dict[str, Any]:
    """Progress of a tagging or upscaling job (kind: tag | upscale; status: queued | running | done | cancelled |
    error)."""
    job = jobs.get(job_id)
    if job is None:
        raise ValueError("job not found")
    return job.to_dict()


@_tool
def list_captions(project_id: str, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    """Final training captions exactly as they will be written to the .txt files."""
    return services.captions(project_id, limit, offset)


@_tool
def get_image(project_id: str, image_id: str) -> dict[str, Any]:
    """Tags, natural caption, rating and final caption of one image."""
    p = services.require_project(project_id)
    from .profiles import normalize_settings

    return services.image_out(services.require_image(image_id, project_id), normalize_settings(p["settings"]))


@_tool
def update_image_caption(project_id: str, image_id: str, tags: list[str] | None = None,
                         nl_caption: str | None = None) -> dict[str, Any]:
    """Replace an image's ordered tag list and/or its natural-language caption."""
    return services.update_image(project_id, image_id, tags, nl_caption)


@_tool
def detect_white_blocks(project_id: str, force: bool = False) -> dict[str, Any]:
    """Find images with white blocks (solid white rectangles used to cover other people). Newly imported images are
    detected automatically; this scans images not checked yet (force=True re-checks all). Returns the ids with blocks.
    To keep the blocks out of the trained LoRA, turn on update_project_settings(settings={"block_tag_auto": true})
    (keyword: settings.block_tag, default "white rectangle") and put that keyword in the negative prompt when
    generating, or remove those images (REST API: POST /api/projects/{project_id}/images/delete)."""
    return services.scan_blocks(project_id, force)


@_tool
def upscale_images(project_id: str, image_ids: list[str] | None = None, min_side: int = 1024, style: str = "art",
                   noise: str = "auto", scale: str = "auto") -> dict[str, Any]:
    """Upscale low-resolution images with waifu2x so the trainer does not blur them, or with scale="1" only remove
    JPEG compression noise (size unchanged). Background job; poll get_job_status. Without image_ids it picks images
    not processed yet: short side below min_side, or for scale="1" every JPEG / lossy WebP image. style: art
    (illustrations / anime) | art_scan (scans) | photo. noise: auto (level 1 for JPEG / lossy WebP, none otherwise) |
    none | 0-3. scale: auto (2x, or 4x when 2x stays below min_side) | 1 (noise reduction only) | 2 | 4. The original is backed up (restore_upscaled_images); tags need not be redone. Images with a short side
    below ~384 px gain little; consider removing them instead."""
    return services.start_upscale(project_id, image_ids, min_side, style, noise, scale)


@_tool
def restore_upscaled_images(project_id: str, image_ids: list[str] | None = None) -> dict[str, Any]:
    """Put the original images back for upscaled images (default: all of them in the project)."""
    return services.restore_upscaled(project_id, image_ids)


@_tool
def bulk_edit_tags(project_id: str, action: str, tags: list[str] | None = None, find: str = "", replace: str = "",
                   image_ids: list[str] | None = None, position: str = "back") -> dict[str, Any]:
    """Bulk edit tags. action: add | remove (supports * wildcard) | replace (find→replace) |
    reapply (regenerate from WD14 scores with current settings) | filter (apply prune groups + blacklist) | clear_nl."""
    return services.bulk_edit(project_id, action, image_ids, tags, find, replace, position)


@_tool
def get_tag_stats(project_id: str, top: int = 100) -> dict[str, Any]:
    """Tag frequency across the dataset plus rating and status counts."""
    stats = services.tag_stats(project_id)
    stats["tags"] = stats["tags"][:top]
    return stats


@_tool
def export_dataset(project_id: str, format: str = "civitai", image_format: str = "original", max_side: int = 0,
                   repeats: int = 10) -> dict[str, Any]:
    """Build the training zip. format: civitai (images + .txt) | kohya | jsonl. Returns a download URL."""
    path, excluded = exporter.export_dataset(project_id, format, image_format, max_side, repeats)
    return {"download_url": services.absolute_url(f"/api/exports/{path.name}"),
            "size_bytes": path.stat().st_size, "excluded": excluded}


@_tool
def get_civitai_training_types() -> dict[str, Any]:
    """Civitai training types with the parameter fields each one accepts, their default values, max batch size and
    documented price (per_step / per_epoch Buzz), plus the default type per base-model profile."""
    return civitai.info()


@_tool
def prepare_civitai_training(project_id: str, training_type: str = "", base_model: str = "", steps: int | None = None,
                             epochs: int | None = None, batch_size: int | None = None, lr: float | None = None,
                             lr_scheduler: str | None = None, optimizer: str | None = None,
                             network_dim: int | None = None, network_alpha: int | None = None,
                             noise_offset: float | None = None, flip_augmentation: bool | None = None,
                             shuffle_tokens: bool | None = None, keep_tokens: int | None = None,
                             trigger_word: str | None = None, min_snr_gamma: int | None = None,
                             train_text_encoder: bool | None = None, text_encoder_lr: float | None = None,
                             continue_from: str = "", sample_prompts: list[str] | None = None,
                             sample_negative: str | None = None, sample_cfg: float | None = None,
                             sample_strength: float | None = None, allow_mature: bool | None = None,
                             only_done: bool = True, priority: str = civitai.DEFAULT_PRIORITY,
                             force_upload: bool = False) -> dict[str, Any]:
    """Upload the project's training images + captions to Civitai and estimate the Buzz cost. Does NOT start training.

    Runs in the background: poll get_civitai_preparation(prep_id) until status is "ready" (or "error").
    training_type: Civitai training type id (sd1, sdxl, anima, flux1-dev, flux1-schnell, flux2klein-4b, flux2klein-9b,
        chroma, ernie, qwen, qwen-2509, qwen21, zimageturbo, zimagebase, boogu, hidream-o1, ideogram4, krea2, mageflow,
        ming, ltx2, ltx23, ltx25, wan-2.1, wan-2.2, minimaxh3); empty = default for the project's base model.
    Fields differ per type — get_civitai_training_types() lists each type's fields, defaults and max_batch:
    base_model only for sd1 / sdxl / anima (AIR / model version ID / civitai.com URL; empty = the type's default);
    trigger_word only for sd1, sdxl, flux1, flux2klein, chroma, zimage*; min_snr_gamma, train_text_encoder and
    text_encoder_lr only for sd1 / sdxl. continue_from: a LoRA of the same type to keep training.
    lr_scheduler: constant | constant_with_warmup | cosine | linear | step. Unset values use Civitai's defaults.
    priority: queue priority — normal (default; the civitai.com trainer's High Priority switch), low (switch off),
    high (API only; effect depends on the account tier). The estimate shows the price for the chosen priority.
    The estimate's cost_full, when set, is the undiscounted price: the final charge may settle at that amount.
    force_upload: re-upload every image instead of reusing earlier uploads (not needed after editing tags — captions
    are always sent fresh). get_civitai_training(...)["request"] holds a past run's settings to train again with.
    """
    params = {"training_type": training_type or None, "base_model": base_model, "steps": steps, "epochs": epochs,
              "batch_size": batch_size, "lr": lr, "lr_scheduler": lr_scheduler, "optimizer": optimizer,
              "network_dim": network_dim, "network_alpha": network_alpha, "noise_offset": noise_offset,
              "flip_augmentation": flip_augmentation, "shuffle_tokens": shuffle_tokens, "keep_tokens": keep_tokens,
              "trigger_word": trigger_word, "min_snr_gamma": min_snr_gamma, "train_text_encoder": train_text_encoder,
              "text_encoder_lr": text_encoder_lr, "continue_from": continue_from, "sample_prompts": sample_prompts,
              "sample_negative": sample_negative, "sample_cfg": sample_cfg, "sample_strength": sample_strength,
              "allow_mature": allow_mature, "only_done": only_done, "max_side": 2048, "priority": priority,
              "force_upload": force_upload}
    return civitai.prepare(project_id, params).to_dict()


@_tool
def get_civitai_preparation(prep_id: str) -> dict[str, Any]:
    """Upload progress, blocked images and (when status is "ready") the estimated cost in Buzz."""
    return civitai.get_prep(prep_id).to_dict()


@_tool
def submit_civitai_training(prep_id: str) -> dict[str, Any]:
    """Start a prepared Civitai training run. This SPENDS the user's Buzz: only call it after telling the user the
    estimated cost from get_civitai_preparation and receiving their explicit confirmation."""
    return civitai.submit(prep_id)


@_tool
def get_civitai_training(workflow_id: str) -> dict[str, Any]:
    """Status of a Civitai training run: moderation, cost, per-epoch LoRA download URLs + sample images, and
    progress (percent, epochs_done / epochs_total, step / total_steps, remaining_seconds) while it runs.
    charged / refunded: Buzz actually charged and refunded (from Civitai's transactions). cancel_requested_at: a
    cancel was sent and Civitai has not stopped the run yet (cancelling takes a few minutes). request: the run's
    settings as prepare_civitai_training parameters, to train again with the same settings."""
    return civitai.refresh_run(workflow_id)


@_tool
def list_active_civitai_trainings() -> list[dict[str, Any]]:
    """Civitai training runs in progress in any project (plus runs finished in the last 24 h) with their progress:
    summary.progress has percent, epochs_done / epochs_total, step / total_steps and remaining_seconds."""
    return civitai.active_runs()


@_tool
def import_lora_to_a1111(workflow_id: str, epoch: int, overwrite: bool = False) -> dict[str, Any]:
    """Import one epoch of a finished Civitai training into A1111 / Forge's Lora folder (needs A1111_URL and Forge
    Neo Chino's LoRA import API). Forge downloads the file itself; the card gets the trigger word, base model type
    and a sample image. Returns name, relative_path and a ready-made prompt like "<lora:name:1> trigger".
    An error mentioning overwrite means a different file with the same name exists: ask the user before retrying
    with overwrite=true."""
    from . import a1111

    return a1111.import_epoch(workflow_id, epoch, overwrite)
