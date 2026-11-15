# MOSS-Image support for Forge Neo
# The negative-prompt block skipping of upstream's guidance, as a Forge setting: the blocks (zero-based) the DiT
# leaves out in the CFG negative pass. Upstream's default is block 22; an empty value is plain CFG.

OPTION = "moss_uncond_skip_layers"
DEFAULT = "22"
LABEL = "[MOSS-Image] Blocks skipped in the negative-prompt pass"
INFO = "zero-based, comma-separated; 22 = upstream's guidance (24 and 26 also help); empty = plain CFG"
# A settings page of its own under the "Stable Diffusion" category, Settings > Stable Diffusion > MOSS-Image, as the
# Qwen-Image 2.1 extension does (options_section splits ("id", "Label", "category") the same way)
SECTION = ("moss_image", "MOSS-Image")
CATEGORY = "sd"


# Forge Neo 2.31+ has MOSS-Image support of its own: who runs the MOSS-Image checkpoints. With "Forge built-in" the
# extension's detection steps aside and Forge's own guess, engine and Skip Layer Guidance take the checkpoint
BACKEND_OPTION = "moss_backend"
EXTENSION = "Extension"
BUILT_IN = "Forge built-in"
BACKEND_LABEL = "[MOSS-Image] Backend"
BACKEND_INFO = ("who runs MOSS-Image checkpoints: this extension, or Forge Neo's built-in support (its own "
                "[Moss-Image] Skip Layer Guidance and [Moss-Image] Reference Method options apply then, in place of "
                "this page's); takes effect at the next generation")


# A LoRA tagged inside one item of a layout acts only inside that item's box (regional_lora.py)
REGIONAL_OPTION = "moss_image_regional_lora"
REGIONAL_LABEL = "[MOSS-Image] Keep a LoRA tagged inside a layout box in that box"
REGIONAL_INFO = ("a <lora:...> in an item's caption acts only inside the item's bbox and on its words; LoRAs in the "
                 "summary or after the layout stay global. Both backends; plain LoRAs only")


# which words of the prompt a LoRA kept in a box acts on: only that box's caption, or the whole prompt (the picture
# stays kept to the box either way)
REGIONAL_TEXT_OPTION = "moss_image_regional_lora_text"
BOX_WORDS = "The box's own words"
WHOLE_PROMPT = "The whole prompt"
REGIONAL_TEXT_LABEL = "[MOSS-Image] Words a LoRA kept in its box acts on"
REGIONAL_TEXT_INFO = ("the picture stays kept to the box either way; the whole prompt gives a style LoRA more strength "
                      "in its box, and lets more of it reach the rest of the picture")


# how a LoRA is kept in its box: one pass of the model per LoRA, each pasted into its boxes (the default: full strength,
# styles too, slower), or inside the model in the same pass (fast; good for LoRAs that change a shape or a thing,
# weak for styles)
REGIONAL_MODE_OPTION = "moss_image_regional_lora_mode"
ONE_PASS = "In one pass"
PER_LORA = "One pass per LoRA"
REGIONAL_MODE_LABEL = "[MOSS-Image] How a LoRA is kept in its box"
REGIONAL_MODE_INFO = ("one pass per LoRA: each box gets its LoRA at full strength (styles too), one more pass of the "
                      "model per LoRA at every step; one pass: as fast as without it, good for LoRAs that change a shape or a "
                      "thing, weak for styles")


def regional_lora_per_lora_pass() -> bool:
    try:
        from modules.shared import opts

        return getattr(opts, REGIONAL_MODE_OPTION, PER_LORA) == PER_LORA
    except Exception:
        return True


def regional_lora_all_text() -> bool:
    try:
        from modules.shared import opts

        return getattr(opts, REGIONAL_TEXT_OPTION, BOX_WORDS) == WHOLE_PROMPT
    except Exception:
        return False


def regional_lora_enabled() -> bool:
    try:
        from modules.shared import opts

        return bool(getattr(opts, REGIONAL_OPTION, True))
    except Exception:  # outside Forge Neo (CPU tests), or before the options exist
        return True


def forge_has_built_in() -> bool:
    try:
        from huggingface_guess import model_list

        return hasattr(model_list, "MOSSImage")
    except Exception:  # outside Forge Neo (CPU tests)
        return False


def use_extension() -> bool:
    # an older Forge Neo has nothing to fall back to: the extension runs the checkpoints whatever the option says
    if not forge_has_built_in():
        return True
    try:
        from modules.shared import opts

        return getattr(opts, BACKEND_OPTION, EXTENSION) != BUILT_IN
    except Exception:  # before the options exist
        return True


def _reload_moss_model() -> None:
    # Forge reloads a checkpoint only when its loading parameters change; a MOSS-Image model already loaded is
    # dropped so that the next generation loads it again with the backend just chosen
    from modules import sd_models

    current = sd_models.model_data.sd_model
    if getattr(current, "is_moss_image", False) or type(current).__name__ == "MOSSImage":
        sd_models.model_data.forge_hash = ""


def parse(text: str, depth: int) -> frozenset[int]:
    """The valid block indices in text; anything else is ignored."""
    layers = set()
    for part in str(text or "").replace(";", ",").replace(" ", ",").split(","):
        if part.strip().lstrip("-").isdigit() and 0 <= int(part) < depth:
            layers.add(int(part))
    return frozenset(layers)


def uncond_skip_layers(depth: int) -> frozenset[int]:
    try:
        from modules.shared import opts

        text = getattr(opts, OPTION, DEFAULT)
    except Exception:  # outside Forge Neo (CPU tests), or before the options exist
        text = DEFAULT
    return parse(text, depth)


def register() -> None:
    from modules import shared
    from modules.options import OptionInfo

    if forge_has_built_in() and BACKEND_OPTION not in shared.opts.data_labels:
        import gradio as gr

        shared.opts.add_option(BACKEND_OPTION, OptionInfo(EXTENSION, BACKEND_LABEL, gr.Radio,
                                                          {"choices": (EXTENSION, BUILT_IN)},
                                                          onchange=_reload_moss_model, section=SECTION,
                                                          category_id=CATEGORY).info(BACKEND_INFO))
    if OPTION not in shared.opts.data_labels:
        shared.opts.add_option(OPTION, OptionInfo(DEFAULT, LABEL, section=SECTION, category_id=CATEGORY,
                                                  infotext="MOSS skip").info(INFO))
    if REGIONAL_OPTION not in shared.opts.data_labels:
        shared.opts.add_option(REGIONAL_OPTION, OptionInfo(True, REGIONAL_LABEL, section=SECTION,
                                                           category_id=CATEGORY).info(REGIONAL_INFO))
    if REGIONAL_MODE_OPTION not in shared.opts.data_labels:
        import gradio as gr

        shared.opts.add_option(REGIONAL_MODE_OPTION, OptionInfo(PER_LORA, REGIONAL_MODE_LABEL, gr.Radio,
                                                                {"choices": (PER_LORA, ONE_PASS)}, section=SECTION,
                                                                category_id=CATEGORY).info(REGIONAL_MODE_INFO))
    if REGIONAL_TEXT_OPTION not in shared.opts.data_labels:
        import gradio as gr

        shared.opts.add_option(REGIONAL_TEXT_OPTION, OptionInfo(BOX_WORDS, REGIONAL_TEXT_LABEL, gr.Radio,
                                                                {"choices": (BOX_WORDS, WHOLE_PROMPT)}, section=SECTION,
                                                                category_id=CATEGORY).info(REGIONAL_TEXT_INFO))
