# MOSS-Image support for Forge Neo
# Regional LoRA in a generation (the arithmetic is in regional.py): before each sampling, the batch's layout prompts
# are read with their <lora:...> tags, each LoRA tagged only inside items gets those items' boxes and text tokens, and
# the linear layers its file touches are wrapped to take its contribution away everywhere else. Everything is a swap
# undone by release(); no Forge file and no class is changed, only this model's own module instances.
#
# Both backends: this extension's DiT (forge_moss.native.transformer) and Forge Neo's built-in one
# (backend.nn.moss_image), which take the same LoRA keys and run the same [image | text] sequence. The text encoders
# differ in how they read the prompt (this extension rebuilds a brace-stripped layout; Forge's unescapes \( and \)),
# and each one's tokens are found the way it reads it.

import logging
import os
import sys
import types
from dataclasses import dataclass, field

import torch

from . import regional, settings

logger = logging.getLogger("forge_moss")

INFOTEXT = "MOSS regional LoRA"
EXTENSION, NATIVE = "extension", "native"
UNCOND = 1  # backend.sampling.sampling_function.UNCOND
# the DiT methods that run MOSS Image Edit's reference images through the blocks: this extension's, Forge's
REFERENCE_BRANCHES = ("reference_kv", "preprocess_refs")


@dataclass
class Spec:
    name: str
    items: list[int]
    strength: float
    layers: dict[str, regional.LowRank] = field(default_factory=dict)  # module path under the DiT -> its low rank


@dataclass
class Layout:
    boxes: list  # per item, (top, left, bottom, right) shares, or None without a bbox
    item_tokens: list[list[int]]  # per item, its text tokens
    loras: list[list[regional.Mention]]  # per item, the LoRAs tagged in it


class _State:
    def __init__(self):
        self.warned: set[str] = set()
        self.reset()

    def reset(self):
        self.specs: list[Spec] = []
        self.layouts: dict[int, Layout] = {}  # text token count -> layout
        self.entries: dict[str, list[tuple[int, regional.LowRank]]] = {}
        self.wrapped: list[tuple[torch.nn.Module, object]] = []  # (module, its own forward before, or None)
        self.dit = None
        self.backend = None
        self.weights: torch.Tensor | None = None  # (specs, B, H*W + T) for the forward pass running
        self.weights_key = None
        self.image_len = 0
        self.in_input_act = False
        self.in_reference = False
        self.all_text = False  # settings.regional_lora_all_text(), read once per generation
        self.multipass = False  # settings.regional_lora_per_lora_pass(), read once per generation


state = _State()
_originals: dict = {}
# low-rank layers per LoRA file, on the CPU, for the session: (filename, mtime, model layout) -> (layers, skipped)
_LOWRANK: dict = {}
_FAST: dict = {}


def _warn_once(key: str, message: str) -> None:
    if key not in state.warned:
        state.warned.add(key)
        logger.warning(f"[MOSS-Image] {message}")


def backend_of(model) -> str | None:
    if getattr(model, "is_moss_image", False):
        return EXTENSION
    if type(model).__name__ == "MOSSImage" and hasattr(model, "text_processing_engine_qwen"):
        return NATIVE
    return None


def other_regional_installed() -> bool:
    """SillySilk's MOSS Regional LoRA does the same on Forge's built-in backend: with both, a LoRA would be taken away
    twice, so this one stands aside."""
    try:
        from modules.shared import opts

        return "moss_regional_lora" in opts.data_labels
    except Exception:
        return False


# ---- the prompt as each text encoder reads it, and its tokens ----


def _fast(tokenizer):
    """A fast tokenizer reports where each token is in the text; Forge loads the slow one for its built-in engine."""
    if getattr(tokenizer, "is_fast", False):
        return tokenizer
    path = tokenizer.name_or_path
    if path not in _FAST:
        from transformers import AutoTokenizer

        _FAST[path] = AutoTokenizer.from_pretrained(path, use_fast=True)
    return _FAST[path]


def tagged_text(prompt: str) -> str:
    """The prompt with its tags, its layout's braces back if sd-dynamic-prompts took them out: to read which LoRA is
    tagged in which item."""
    from .layout import restore_layout_json

    return restore_layout_json(prompt.strip())


def encoded(model, backend: str, prompt: str):
    """(items, token count, locate): the layout's items in the text the encoder reads (no tags), how many tokens it
    makes, and locate(spans) giving the tokens inside each (start, end) span of the items' text. None when the tokens
    cannot be located exactly."""
    from .layout import restore_layout_json

    stripped = regional.strip_networks(prompt)
    if backend == EXTENSION:
        from .text_engine import MAX_LENGTH, TEMPLATE

        engine = model.text_engine_moss
        body = restore_layout_json(stripped.strip())
        prefix = TEMPLATE.index("{}")
        encoding = _fast(engine.tokenizer.tokenizer)(TEMPLATE.format(body), add_special_tokens=False,
                                                     return_offsets_mapping=True)
        ids = list(encoding["input_ids"])[:MAX_LENGTH]
        if ids != list(engine.token_ids(stripped)):
            _warn_once("tokens", "regional LoRA: the prompt's tokens could not be located; the LoRAs stay global")
            return None
        offsets = encoding["offset_mapping"][:len(ids)]
        return regional.parse_items(body), len(ids), lambda spans: regional.token_spans(
            offsets, [(prefix + a, prefix + b) for a, b in spans])

    # Forge's built-in engine reads the prompt as it comes: a layout sd-dynamic-prompts stripped keeps no braces, so
    # its items are found on the layout rebuilt and placed back on the text the model reads (only braces differ)
    engine = model.text_processing_engine_qwen
    body = stripped.strip() or " "
    items = regional.parse_items(body)
    to_body = None
    if items is None:
        rebuilt = restore_layout_json(body)
        items = regional.parse_items(rebuilt)
        to_body = regional.insertion_map(body, rebuilt)
    template = engine.llama_template
    prefix = template.index("{}")
    escaped, index = regional.unescape_with_map(template.format(body))
    slow = engine.tokenizer.tokenizer
    encoding = _fast(slow)(escaped, return_offsets_mapping=True)
    if list(encoding["input_ids"]) != list(slow(escaped)["input_ids"]):
        _warn_once("tokens", "regional LoRA: the prompt's tokens could not be located; the LoRAs stay global")
        return None
    offsets = encoding["offset_mapping"]

    def locate(spans):
        if to_body is not None:
            spans = [(to_body[a], to_body[b]) for a, b in spans]
        return regional.token_spans(offsets, [(index[prefix + a], index[prefix + b]) for a, b in spans])

    return items, len(offsets), locate


def collect(model, backend: str, prompts, width: int, height: int) -> tuple[dict[int, Layout], set[str]]:
    """The layouts of the batch's prompts by token count, and the LoRAs tagged outside any item."""
    layouts: dict[int, Layout] = {}
    global_names: set[str] = set()
    for prompt in dict.fromkeys(prompts):
        tagged = tagged_text(prompt)
        tagged_items = regional.parse_items(tagged)
        if not tagged_items or not any(item.loras for item in tagged_items):
            continue
        found = encoded(model, backend, prompt)
        if found is None:
            continue
        items, count, locate = found
        if items is None or len(items) != len(tagged_items):
            _warn_once("items", "regional LoRA: the layout changed when its LoRA tags were removed; the LoRAs stay global")
            continue
        layout = Layout([regional.box_fraction(it.bbox, width, height) if it.bbox else None for it in items],
                        locate([(it.start, it.end) for it in items]), [it.loras for it in tagged_items])
        if count in layouts and layouts[count] != layout:
            _warn_once("ambiguous", "regional LoRA: two prompts of the batch have the same length and different "
                                    "layouts; the first one is used")
            continue
        layouts[count] = layout
        global_names |= regional.global_names(tagged, tagged_items)
    return layouts, global_names


# ---- the LoRAs' low-rank layers ----


def _key_map(unet_model) -> dict:
    # the key map sd_forge_lora uses, with this extension's aliases (lora.py wraps it)
    networks = sys.modules.get("networks")
    keys = getattr(networks, "model_lora_keys_unet", None)
    if keys is None:
        from backend.patcher.lora import model_lora_keys_unet as keys
    return keys(unet_model)


def _out_features(dit, path: str, parts) -> int:
    try:
        return int(dit.get_submodule(path).weight.shape[0])
    except Exception:
        return max(lr.up.shape[0] if offset is None else offset[1] + offset[2] for offset, lr in parts)


def lowrank_layers(filename: str, unet_model) -> tuple[dict[str, regional.LowRank], int]:
    """Per module path under the DiT, the LoRA file's plain low-rank pair at strength 1; and how many of its layers are
    not plain LoRA (they stay global)."""
    dit = unet_model.diffusion_model
    fused = any(getattr(layer, "fused_qkv", False) for layer in getattr(dit, "layers", [])[:1])
    key = (filename, os.path.getmtime(filename), type(dit).__name__, fused)
    if key in _LOWRANK:
        return _LOWRANK[key]
    from backend.patcher.lora import load_lora
    from backend.state_dict import state_dict_prefix_replace
    from backend.utils import load_torch_file

    sd = load_torch_file(filename, safe_load=True)
    if any(k.startswith("lora_unet__") for k in sd):
        sd = state_dict_prefix_replace(sd, {"lora_unet__": "lora_unet_"})
    patches = load_lora(sd, _key_map(unet_model))
    if isinstance(patches, tuple):  # (patch_dict, unmatched) in some Forge versions
        patches = patches[0]
    pieces: dict[str, list] = {}
    skipped = 0
    for model_key, adapter in patches.items():
        # a tuple key is (fused weight, (dim, start, size)): a q, k or v LoRA on the fused qkv projection
        name, offset = model_key if isinstance(model_key, tuple) else (model_key, None)
        weights = getattr(adapter, "weights", None)
        if not (isinstance(name, str) and name.startswith("diffusion_model.") and name.endswith(".weight")) or weights is None:
            skipped += 1
            continue
        lr = regional.extract_lowrank(weights)
        if lr is None or (offset is not None and offset[0] != 0):
            skipped += 1
            continue
        pieces.setdefault(name[len("diffusion_model."):-len(".weight")], []).append((offset, lr))
    del sd

    layers = {path: regional.combine(parts, _out_features(dit, path, parts)) for path, parts in pieces.items()}
    _LOWRANK[key] = (layers, skipped)
    return layers, skipped


def _lora_file(name: str) -> str | None:
    networks = sys.modules.get("networks")
    if networks is None:
        return None
    on_disk = networks.available_network_aliases.get(name) or networks.available_networks.get(name)
    return getattr(on_disk, "filename", None)


def build_specs(layouts: dict[int, Layout], global_names: set[str], unet_model) -> list[Spec]:
    layout = next(iter(layouts.values()))
    per_name: dict[str, tuple[list[int], float]] = {}
    for i, mentions in enumerate(layout.loras):
        for m in mentions:
            items, strength = per_name.get(m.name, ([], 0.0))
            # Forge loads a LoRA once per mention: two mentions are twice its strength
            per_name[m.name] = (items if i in items else [*items, i], strength + m.unet)

    specs = []
    for name, (items, strength) in per_name.items():
        if name in global_names:
            logger.info(f"[MOSS-Image] LoRA '{name}' is also outside the boxes: it stays global")
            continue
        if any(layout.boxes[i] is None for i in items):
            _warn_once(f"nobox:{name}", f"LoRA '{name}' is in an item without a bbox: it stays global")
            continue
        if strength == 0:
            continue
        filename = _lora_file(name)
        if filename is None:
            _warn_once(f"missing:{name}", f"LoRA '{name}' was not found: it stays global")
            continue
        layers, skipped = lowrank_layers(filename, unet_model)
        if skipped:
            _warn_once(f"skipped:{name}", f"LoRA '{name}': {skipped} layer(s) are not plain LoRA and stay global")
        if not layers:
            _warn_once(f"empty:{name}", f"LoRA '{name}' has no layer that can be kept in its box: it stays global")
            continue
        specs.append(Spec(name, items, strength,
                          {path: regional.LowRank(lr.up, lr.down, lr.scale * strength) for path, lr in layers.items()}))
    return specs


# ---- the forward passes ----


def _rows_cond(transformer_options, rows: int) -> list[bool]:
    chunks = (transformer_options or {}).get("cond_or_uncond") or []
    if not chunks or rows % len(chunks):
        return [True] * rows
    per = rows // len(chunks)
    return [chunks[r // per] != UNCOND for r in range(rows)]


def _references_in_sequence() -> bool:
    """Forge's built-in backend appends reference images to the sequence with its "index" reference methods: the
    image and text tokens then no longer line up with the weights."""
    if state.backend != NATIVE:
        return False
    from backend.args import dynamic_args

    return bool(dynamic_args.ref_latents) and getattr(dynamic_args, "moss_reference_method", None) in (
        "index", "index_timestep_zero", None)


def _set_weights(x, context, transformer_options) -> None:
    b, _, h, w = x.shape
    text_len = context.shape[1]
    rows = _rows_cond(transformer_options, b)
    key = (b, h, w, text_len, tuple(rows), x.device, x.dtype)
    if key == state.weights_key:
        return
    layout = state.layouts.get(text_len) if any(rows) else None
    if any(rows) and layout is None:
        _warn_once(f"len{text_len}", "regional LoRA: no layout has this prompt's length; the LoRAs act only in their "
                                     "boxes, not on their text")
    boxes = (layout or next(iter(state.layouts.values()))).boxes
    weights = regional.token_weights([s.items for s in state.specs], boxes, layout.item_tokens if layout else None,
                                     rows, h, w, text_len, x.device, all_text=state.all_text)
    state.weights, state.weights_key, state.image_len = weights.to(x.dtype), key, h * w


def _dit_forward(original):
    def forward(x, timestep, *args, **kwargs):
        context = kwargs.get("context", args[0] if args else None)
        options = kwargs.get("transformer_options", args[1] if len(args) > 1 else None)
        if not state.specs or context is None or x.dim() != 4:
            return original(x, timestep, *args, **kwargs)
        if _references_in_sequence():
            _warn_once("refs", "regional LoRA is off while reference images are in the sequence")
            state.weights_key, state.weights = None, None
            return original(x, timestep, *args, **kwargs)
        if state.multipass:
            return _per_lora(original, x, timestep, context, options, args, kwargs)
        # kept from one step to the next while the batch's shape and its cond / uncond rows stay the same
        _set_weights(x, context, options)
        return original(x, timestep, *args, **kwargs)

    return forward


def _per_lora(original, x, timestep, context, options, args, kwargs):
    """One pass per LoRA mode: the DiT runs once with no regional LoRA, then once per LoRA with that one on the whole
    picture and the others off; each LoRA's result is taken inside its boxes, softened at their edges. As strong in its
    box as the LoRA on the whole picture, and nothing of it outside; one more pass per LoRA, every step."""
    b, _, h, w = x.shape
    text_len = context.shape[1]
    specs = state.specs
    state.image_len, state.weights_key = h * w, None

    def run(on):
        weights = torch.zeros(len(specs), b, h * w + text_len, dtype=x.dtype, device=x.device)
        if on is not None:
            weights[on] = 1
        state.weights = weights
        return original(x, timestep, *args, **kwargs)

    try:
        base = run(None)
        passes = [run(s) for s in range(len(specs))]
    finally:
        state.weights = None
    rows = _rows_cond(options, b)
    layout = (state.layouts.get(text_len) if any(rows) else None) or next(iter(state.layouts.values()))
    masks = regional.region_masks([s.items for s in specs], layout.boxes, h, w, device=x.device)
    return regional.composite(base, passes, masks)


def _weights_for(x: torch.Tensor) -> torch.Tensor | None:
    weights = state.weights
    if weights is None or state.in_reference or x.dim() != 3 or x.shape[0] != weights.shape[1]:
        return None
    if x.shape[1] == weights.shape[-1]:
        return weights
    if x.shape[1] == weights.shape[-1] - state.image_len:  # the text embedder: text tokens only
        return weights[..., state.image_len:]
    if x.shape[1] == state.image_len:  # the video embedder and the final layer: image tokens only
        return weights[..., :state.image_len]
    return None


def _reference_branch(original):
    # MOSS Image Edit's reference images run through the blocks on their own (reference_kv here, preprocess_refs in
    # Forge's backend): a reference the size of the picture must not be taken for it, so the LoRAs stay whole there
    def run(*args, **kwargs):
        state.in_reference = True
        try:
            return original(*args, **kwargs)
        finally:
            state.in_reference = False

    return run


def _apply(path, out, x):
    entries = state.entries.get(path)
    weights = _weights_for(x) if entries else None
    if weights is None:
        return out
    pairs = []
    for i, lr in entries:
        if lr.up.device != x.device or lr.up.dtype != x.dtype:
            moved = lr.to(x.device, x.dtype)
            lr.up, lr.down = moved.up, moved.down  # the spec's own copy: safe to cast in place
        pairs.append((lr, weights[i]))
    return regional.subtract_outside(out, x, pairs)


def _linear_forward(module, before):
    call = before if before is not None else types.MethodType(type(module).forward, module)

    def forward(x, *args, **kwargs):
        out = call(x, *args, **kwargs)
        if state.in_input_act:  # inside linear_input_act, which takes the LoRA away itself (_input_act)
            return out
        return _apply(getattr(module, "_moss_regional_path", None), out, x)

    return forward


def _input_act(original, eager):
    """linear(act(x)) with the activation fused into the matmul never calls the module's forward (the INT8 path of
    Forge's linear_input_act): the LoRA is taken away from its output, on the activation computed eagerly."""

    def linear_input_act(linear, x, input_act, *args, **kwargs):
        path = getattr(linear, "_moss_regional_path", None)
        state.in_input_act = True
        try:
            out = original(linear, x, input_act, *args, **kwargs)
        finally:
            state.in_input_act = False
        if path is None or path not in state.entries or _weights_for(x) is None:
            return out
        return _apply(path, out, eager(x, input_act, *args, **kwargs))

    linear_input_act._moss_regional = True
    return linear_input_act


def _hook_input_act(backend: str) -> None:
    if backend == EXTENSION:
        from . import kernels, transformer

        original = kernels._forge_linear_input_act
        if original is None:  # no fused path: ffn_down goes through the module's forward
            return

        def eager(x, input_act, *args, **kwargs):
            hidden, gate = x.chunk(2, -1)
            return transformer.situ(hidden, gate)

        _originals["kernels"] = original
        kernels._forge_linear_input_act = _input_act(original, eager)
    else:
        from backend.nn import moss_image
        from backend.operations_mixed_precision import _eager_input_act

        def eager(x, input_act, act_weight=None, act_eps=0.0, *args, **kwargs):
            return _eager_input_act(x, input_act, act_weight, act_eps)

        _originals["native"] = moss_image.linear_input_act
        moss_image.linear_input_act = _input_act(moss_image.linear_input_act, eager)


def _unhook_input_act() -> None:
    if "kernels" in _originals:
        from . import kernels

        kernels._forge_linear_input_act = _originals.pop("kernels")
    if "native" in _originals:
        from backend.nn import moss_image

        moss_image.linear_input_act = _originals.pop("native")


def patch(dit, specs: list[Spec], layouts: dict[int, Layout], backend: str, device=None, dtype=None) -> None:
    """Wraps the DiT's forward and the layers the specs touch. device and dtype: where to place the low-rank matrices
    before Forge plans the model's memory (placing them inside the forward could overfill the GPU)."""
    release()
    entries: dict[str, list] = {}
    for i, spec in enumerate(specs):
        for path, lr in spec.layers.items():
            entries.setdefault(path, []).append((i, lr))
    missing = 0
    for path in list(entries):
        try:
            module = dit.get_submodule(path)
        except AttributeError:
            missing += 1
            del entries[path]
            continue
        before = module.__dict__.get("forward")
        module._moss_regional_path = path
        module.forward = _linear_forward(module, before)
        state.wrapped.append((module, before))
    if missing:
        _warn_once("paths", f"regional LoRA: {missing} LoRA layer(s) are not in this model and stay global")
    if device is not None:
        for spec in specs:
            for lr in spec.layers.values():
                moved = lr.to(device, dtype or lr.up.dtype)
                lr.up, lr.down = moved.up, moved.down
    state.specs, state.layouts, state.entries, state.backend, state.dit = specs, layouts, entries, backend, dit
    for name in ("forward", *REFERENCE_BRANCHES):
        if name == "forward" or hasattr(type(dit), name):
            before = dit.__dict__.get(name)
            _originals[f"dit.{name}"] = before
            bound = before or types.MethodType(getattr(type(dit), name), dit)
            setattr(dit, name, _dit_forward(bound) if name == "forward" else _reference_branch(bound))
    _hook_input_act(backend)


def release() -> None:
    """Undoes every swap of patch()."""
    for module, before in state.wrapped:
        if before is None:
            module.__dict__.pop("forward", None)
        else:
            module.forward = before
        module.__dict__.pop("_moss_regional_path", None)
    if state.dit is not None:
        for key in [k for k in _originals if k.startswith("dit.")]:
            name, before = key[len("dit."):], _originals.pop(key)
            if before is None:
                state.dit.__dict__.pop(name, None)
            else:
                setattr(state.dit, name, before)
    try:
        _unhook_input_act()
    finally:
        state.reset()


def describe(specs: list[Spec]) -> str:
    return "; ".join(f"{s.name}: box {'+'.join(str(i + 1) for i in s.items)}" for s in specs)


def prepare(p) -> None:
    """Before each sampling (first pass and hires fix): the batch's LoRAs that are tagged only inside boxes are kept
    there."""
    release()
    model = getattr(p, "sd_model", None)
    backend = backend_of(model)
    if backend is None or not settings.regional_lora_enabled():
        return
    if other_regional_installed():
        _warn_once("other", "MOSS Regional LoRA (another extension) is installed: it keeps the LoRAs in their boxes "
                            "instead of this extension")
        return
    if getattr(p, "_ad_inner", False):  # ADetailer's crops: the boxes no longer line up
        return
    hires = getattr(p, "is_hr_pass", False)
    source = p.all_hr_prompts if hires and getattr(p, "all_hr_prompts", None) else p.all_prompts
    start = p.iteration * p.batch_size
    prompts = source[start:start + p.batch_size] or source
    try:
        # the boxes are pixels of the first pass's size; the hires fix keeps its shape, so the shares hold
        layouts, global_names = collect(model, backend, prompts, p.width, p.height)
        if not layouts:
            return
        unet = model.forge_objects.unet
        specs = build_specs(layouts, global_names, unet.model)
        if not specs:
            return
        from backend import memory_management

        patch(unet.model.diffusion_model, specs, layouts, backend, memory_management.get_torch_device(),
              getattr(unet.model, "computation_dtype", None))
        state.all_text = settings.regional_lora_all_text()
        state.multipass = settings.regional_lora_per_lora_pass()
        p.extra_generation_params[INFOTEXT] = (describe(specs) + ("; one pass per LoRA" if state.multipass else "")
                                               + ("; whole prompt" if state.all_text and not state.multipass else ""))
        for spec in specs:
            logger.info(f"[MOSS-Image] LoRA '{spec.name}' kept in box {[i + 1 for i in spec.items]} "
                        f"({len(spec.layers)} layers)")
    except Exception:
        logger.exception("[MOSS-Image] regional LoRA failed; the LoRAs stay global")
        release()
