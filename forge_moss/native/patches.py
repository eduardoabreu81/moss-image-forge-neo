# MOSS-Image support for Forge Neo
# Runtime hooks into Forge Neo's loader and sampler. Every hook hands anything that is not MOSS-Image to the original
# function untouched; Forge Neo itself stays unchanged. Follows the FLUX 3 extension's patches.py.

import functools
import logging

import torch
from backend import loader
from backend.nn import krea
from huggingface_guess import detection, model_list

from . import edit, model, settings, text_encoder
from .checkpoints import check_vae, clean_transformer_state_dict, text_encoder_prefix
from .engine import MossImageEngine
from .transformer import MossImageDiT

logger = logging.getLogger("forge_moss")

_applied = False


def _hook_detection() -> None:
    original = detection.detect_unet_config

    def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        # with "[MOSS-Image] Backend" on Forge built-in, the checkpoint is left to Forge Neo's own MOSS-Image guess
        dit_config = model.detect(state_dict, key_prefix) if settings.use_extension() else None
        if dit_config is not None:
            return dit_config
        return original(state_dict, key_prefix, *args, **kwargs)

    detection.detect_unet_config = detect_unet_config


def _hook_components() -> None:
    original = loader.load_huggingface_component

    # wraps leaves __wrapped__ behind, which inspect.getsource follows: the compat checks of the other extensions that
    # read Forge's loader source still reach it through this hook
    @functools.wraps(original)
    def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        if not isinstance(guess, model.MossImage):
            return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

        if cls_name == "MOSSImageTransformer2DModel":
            if isinstance(state_dict, dict):
                clean_transformer_state_dict(state_dict)
            # the Krea 2 branch is Forge's generic DiT load: dtype, quantization (INT8 ConvRot) and device handling;
            # it imports backend.nn.krea.SingleStreamDiT inside the function, so the swap routes the build to us
            original_krea = krea.SingleStreamDiT
            krea.SingleStreamDiT = MossImageDiT
            try:
                return original(guess, component_name, lib_name, "Krea2Transformer2DModel", repo_path, state_dict)
            finally:
                krea.SingleStreamDiT = original_krea
        if cls_name == text_encoder.LOADER_CLASS:
            return text_encoder.load(original, guess, component_name, lib_name, repo_path, state_dict)
        if cls_name == "AutoencoderKLFlux2":
            check_vae(state_dict)

        return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

    loader.load_huggingface_component = load_huggingface_component


def _hook_text_encoder_file() -> None:
    # Forge sorts the files selected under VAE / Text Encoder by their keys and knows no Qwen3.5: a Qwen3.5 file goes
    # where the MOSS-Image guess looks for its text encoder (model.clip_target)
    original = loader.replace_state_dict

    def replace_state_dict(sd, asd, *rest):
        # Forge Neo calls (sd, asd, guess, path), and (sd, asd, metadata, guess, path) once its GGUF update lands
        guess = rest[-2]
        if isinstance(guess, model.MossImage):
            try:
                text_encoder_prefix(asd)
            except ValueError:
                pass
            else:
                prefix = f"{guess.text_encoder_key_prefix[0]}{model.TEXT_ENCODER}.transformer."
                for key in [k for k in sd if k.startswith(prefix)]:
                    del sd[key]
                sd.update({f"{prefix}{k}": v for k, v in asd.items()})
                return sd
        return original(sd, asd, *rest)

    loader.replace_state_dict = replace_state_dict


def _is_moss_image() -> bool:
    from modules import shared

    return bool(getattr(shared.sd_model, "is_moss_image", False))


def _hook_cond_concat() -> None:
    # Forge batches the positive and negative prompts together by repeating the shorter text up to a common length;
    # the MOSS-Image DiT attends to every text token, so a repeated prompt is a different prompt. Conditionings of
    # different lengths run separately instead (upstream encodes each sample at its own length)
    from backend.sampling.condition import ConditionCrossAttn

    original = ConditionCrossAttn.can_concat

    def can_concat(self, other):
        if self.cond.shape != other.cond.shape and _is_moss_image():
            return False
        return original(self, other)

    ConditionCrossAttn.can_concat = can_concat


def _hook_can_reference() -> None:
    # ImageStitch (Forge's built-in multi-image input) only encodes references for the models dynamic_args.can_reference
    # names; with the edit setting on, this one too
    from backend.args import dynamic_args

    original = getattr(dynamic_args, "can_reference", None)
    if original is None or getattr(original, "_moss_image", False):
        return

    def can_reference():
        return edit.enabled() or original()

    can_reference._moss_image = True
    dynamic_args.can_reference = can_reference


def register_callbacks() -> None:
    """Per script load: a UI reload clears script_callbacks and runs scripts/moss_image.py again."""
    from modules import script_callbacks

    script_callbacks.on_cfg_denoiser(edit.on_cfg_denoiser, name="moss_image_edit")


def _register_scheduler() -> None:
    from modules import sd_schedulers, shared

    from . import scheduler as moss_scheduler

    if moss_scheduler.LABEL in sd_schedulers.schedulers_map:
        return
    entry = sd_schedulers.Scheduler(moss_scheduler.NAME, moss_scheduler.LABEL, moss_scheduler.sigmas,
                                    aliases=list(moss_scheduler.ALIASES))
    sd_schedulers.all_schedulers.append(entry)
    if entry.label not in shared.opts.hide_schedulers:
        sd_schedulers.schedulers.append(entry)
    sd_schedulers.schedulers_map.update({entry.name: entry, entry.label: entry,
                                         **{alias: entry for alias in (entry.aliases or [])}})


def apply() -> None:
    global _applied
    if _applied:
        return

    _hook_detection()
    if model.MossImage not in model_list.models:
        model_list.models.append(model.MossImage)
    if MossImageEngine not in loader.possible_models:
        loader.possible_models = (*loader.possible_models, MossImageEngine)
    _hook_components()
    _hook_text_encoder_file()
    _hook_cond_concat()
    _hook_can_reference()
    try:
        _register_scheduler()
    except Exception as e:
        # the model still works: Simple gives the same shifted curve, without the training-grid snapping
        logger.warning(f"[MOSS-Image] could not add the MOSS scheduler: {e}")
    try:
        from . import settings

        settings.register()
    except Exception as e:
        # the DiT falls back to upstream's default, block 22
        logger.warning(f"[MOSS-Image] could not add its setting: {e}")
    try:
        edit.register_settings()
    except Exception as e:
        # txt2img and img2img still work; editing stays off
        logger.warning(f"[MOSS-Image] could not add the MOSS Image Edit settings: {e}")
    # Forge's built-in LoRA extension loads before this one; when it does not, the engine retries at model load
    from . import lora

    lora.install()
    try:
        from . import presets

        presets.register()
    except Exception as e:
        # the model still works: Euler, MOSS, 50 steps, CFG 3 by hand
        logger.warning(f"[MOSS-Image] could not add the moss UI preset: {e}")

    _applied = True
    print(f"[MOSS-Image] native backend enabled (torch {torch.__version__})")
