# MOSS-Image support for Forge Neo
# Regional LoRA, the parts that need no Forge: a LoRA tagged inside one item of a layout prompt acts only inside that
# item's box (and on that item's own text tokens), and a LoRA tagged anywhere else stays global.
#
# The layout already places each item with its bbox, and MOSS-Image runs [image tokens | text tokens] through one
# sequence, so no attention bias is needed. Forge merges (or applies on the fly) every LoRA into the linear layers:
#     full(x) = base(x) + sum_r delta_r(x),    delta_r(x) = (x @ down_r^T) @ up_r^T * scale_r
# and taking LoRA r away from the tokens outside its region is
#     out = full(x) - (1 - w_r[token]) * delta_r(x)
# with w_r = 1 on the image tokens inside its items' boxes and on those items' text tokens, 0 elsewhere.
#
# The approach is SillySilk's MOSS Regional LoRA (Forge-Neo-Silk-Custom, AGPL-3.0), itself built on the Forge Couple
# fork's Krea 2 engine; written again here for this extension's two backends, with sd-dynamic-prompts' stripped
# layouts read through layout.restore_layout_json.

import json
import math
import re
from dataclasses import dataclass, field

import torch

# Forge's extra network syntax (modules/extra_networks.re_extra_net): <type:name[:args...]>
NETWORK = re.compile(r"<(\w+):([^>]+)>")


@dataclass(frozen=True)
class Mention:
    name: str
    unet: float  # the strength Forge loads the LoRA at, for the model (not the text encoder)


@dataclass
class Item:
    start: int  # the span of the item's {...} in the text
    end: int
    bbox: tuple[float, float, float, float] | None  # (y1, x1, y2, x2), pixels from the picture's centre
    loras: list[Mention] = field(default_factory=list)


def mentions(text: str, default: float = 1.0) -> list[Mention]:
    """The <lora:...> tags in text, with the unet strength Forge gives each (extra_networks_lora.activate): the 2nd
    argument is the text encoder's, the 3rd the model's (else the 2nd), te= and unet= override them."""
    found = []
    for kind, args in NETWORK.findall(str(text or "")):
        if kind != "lora":
            continue
        parts = args.split(":")
        positional = [p for p in parts if "=" not in p]
        named = dict(p.split("=", 1) for p in parts if "=" in p)
        if not positional or not positional[0]:
            continue
        try:
            te = (0.0 if "@" in positional[1] else float(positional[1])) if len(positional) > 1 else float(default)
            te = float(named.get("te", te))
            unet = float(positional[2]) if len(positional) > 2 else te
            unet = float(named.get("unet", unet))
        except ValueError:
            continue
        found.append(Mention(positional[0], unet))
    return found


def strip_networks(text: str) -> str:
    """The text as Forge hands it to the text engine: every extra network tag removed in place."""
    return NETWORK.sub("", str(text or ""))


class _Scanner:
    """A JSON reader that keeps where each value starts and ends. Objects are dicts of key -> (value, start, end),
    lists are lists of (value, start, end)."""

    def __init__(self, text: str, at: int = 0):
        self.s = text
        self.i = at

    def space(self):
        while self.i < len(self.s) and self.s[self.i] in " \t\r\n":
            self.i += 1

    def value(self):
        self.space()
        start = self.i
        c = self.s[self.i]
        if c == "{":
            v = self.obj()
        elif c == "[":
            v = self.arr()
        elif c == '"':
            v = self.string()
        else:
            v = self.scalar()
        return v, start, self.i

    def obj(self):
        out = {}
        self.i += 1
        self.space()
        if self.s[self.i] == "}":
            self.i += 1
            return out
        while True:
            self.space()
            key = self.string()
            self.space()
            if self.s[self.i] != ":":
                raise ValueError(f"expected ':' at {self.i}")
            self.i += 1
            out[key] = self.value()
            self.space()
            c = self.s[self.i]
            self.i += 1
            if c == "}":
                return out
            if c != ",":
                raise ValueError(f"expected ',' or '}}' at {self.i - 1}")

    def arr(self):
        out = []
        self.i += 1
        self.space()
        if self.s[self.i] == "]":
            self.i += 1
            return out
        while True:
            out.append(self.value())
            self.space()
            c = self.s[self.i]
            self.i += 1
            if c == "]":
                return out
            if c != ",":
                raise ValueError(f"expected ',' or ']' at {self.i - 1}")

    def string(self):
        if self.s[self.i] != '"':
            raise ValueError(f"expected a string at {self.i}")
        start = self.i
        self.i += 1
        while self.s[self.i] != '"':
            self.i += 2 if self.s[self.i] == "\\" else 1
        self.i += 1
        try:
            return json.loads(self.s[start:self.i])
        except ValueError:  # an escape JSON does not know, such as Forge's \( : the raw text is enough here
            return self.s[start + 1:self.i - 1]

    _SCALAR = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|true|false|null")

    def scalar(self):
        m = self._SCALAR.match(self.s, self.i)
        if not m:
            raise ValueError(f"unexpected character at {self.i}")
        self.i = m.end()
        token = m.group(0)
        return {"true": True, "false": False, "null": None}.get(token, float(token) if token[0] in "-0123456789" else token)


def parse_items(text: str) -> list[Item] | None:
    """The items of the layout the text starts with (after any whitespace), or None when it does not start with one."""
    text = str(text or "")
    start = len(text) - len(text.lstrip())
    if not text[start:].startswith("{"):
        return None
    try:
        root, _, _ = _Scanner(text, start).value()
    except (ValueError, IndexError):
        return None
    if not isinstance(root, dict) or "summary" not in root or "items" not in root or not isinstance(root["items"][0], list):
        return None
    items = []
    for value, s, e in root["items"][0]:
        if not isinstance(value, dict):
            return None
        bbox = None
        if "bbox" in value and isinstance(value["bbox"][0], list) and len(value["bbox"][0]) == 4:
            numbers = [v for v, _, _ in value["bbox"][0]]
            if all(isinstance(n, float) for n in numbers):
                bbox = tuple(numbers)
        items.append(Item(s, e, bbox, mentions(text[s:e])))
    return items


def global_names(text: str, items: list[Item]) -> set[str]:
    """The LoRAs tagged outside every item (the summary, the words after the layout): those stay global."""
    outside, last = [], 0
    for item in items:
        outside.append(text[last:item.start])
        last = item.end
    outside.append(text[last:])
    return {m.name for m in mentions("".join(outside))}


def box_fraction(bbox, width: int, height: int) -> tuple[float, float, float, float]:
    """(y1, x1, y2, x2) pixels from the centre -> (top, left, bottom, right) shares of the picture, as the layout's
    abs_centralized boxes are written: minus height // 2 and width // 2."""
    y1, x1, y2, x2 = bbox
    oy, ox = height // 2, width // 2
    clamp = lambda v: min(max(v, 0.0), 1.0)  # noqa: E731
    top, bottom = sorted((clamp((y1 + oy) / height), clamp((y2 + oy) / height)))
    left, right = sorted((clamp((x1 + ox) / width), clamp((x2 + ox) / width)))
    return top, left, bottom, right


def unescape_with_map(text: str) -> tuple[str, list[int]]:
    """Forge's prompt parser turns \\( and \\) into ( and ) (Forge Neo's built-in MOSS-Image engine reads its prompt
    through it). The text the tokenizer sees and, for every index of text plus one past its end, its index there."""
    out, index = [], []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1] in "()":
            index += [len(out), len(out)]
            out.append(text[i + 1])
            i += 2
        else:
            index.append(len(out))
            out.append(text[i])
            i += 1
    index.append(len(out))
    return "".join(out), index


def insertion_map(original: str, rebuilt: str, inserted: str = "{}") -> list[int]:
    """For every index of rebuilt plus one past its end, the index in original it stands for, where rebuilt is
    original with characters of `inserted` added (layout.restore_layout_json puts back the braces sd-dynamic-prompts
    took out). Past the first other difference (restore_layout_json tidies the words after the layout), indices go to
    original's end."""
    out, j = [], 0
    diverged = False
    for c in rebuilt:
        if not diverged and j < len(original) and c == original[j]:
            out.append(j)
            j += 1
        elif not diverged and c in inserted:
            out.append(j)
        else:
            diverged = True
            out.append(len(original))
    out.append(len(original) if diverged else j)
    return out


def token_spans(offsets, char_spans) -> list[list[int]]:
    """For each (start, end) span of characters, the tokens wholly inside it: a token that straddles two items, such
    as '},{', belongs to neither."""
    return [[t for t, (a, b) in enumerate(offsets) if a >= cs and b <= ce and b > a] for cs, ce in char_spans]


# ---- the token weights of one forward pass ----


def box_mask(box, height: int, width: int, device=None) -> torch.Tensor:
    """(height * width,) 1 on the latent cells a (top, left, bottom, right) box covers, at least one cell."""
    top, left, bottom, right = box
    mask = torch.zeros(height, width, dtype=torch.float32, device=device)
    r0 = min(math.floor(top * height), height - 1)
    c0 = min(math.floor(left * width), width - 1)
    r1 = max(math.ceil(bottom * height), r0 + 1)
    c1 = max(math.ceil(right * width), c0 + 1)
    mask[r0:r1, c0:c1] = 1.0
    return mask.flatten()


def token_weights(spec_items, boxes, item_tokens, rows_cond, height, width, text_len, device=None,
                  all_text=False) -> torch.Tensor:
    """(specs, rows, height * width + text_len): where each regional LoRA keeps acting.

    spec_items: per LoRA, the indices of its items; boxes: per item, its (top, left, bottom, right); item_tokens: per
    item, its text tokens, or None when the prompt of this pass is not the layout (then only the image tokens are
    weighted); rows_cond: per row of the batch, True for the positive prompt (only its rows get the items' text).
    all_text: the LoRA acts on every token of the positive prompt, and only the image is kept to the boxes.
    """
    image = height * width
    out = torch.zeros(len(spec_items), len(rows_cond), image + text_len, dtype=torch.float32, device=device)
    for s, items in enumerate(spec_items):
        region = torch.zeros(image, dtype=torch.float32, device=device)
        for i in items:
            region = torch.maximum(region, box_mask(boxes[i], height, width, device))
        out[s, :, :image] = region
        if all_text and items:
            for r, cond in enumerate(rows_cond):
                if cond:
                    out[s, r, image:] = 1.0
            continue
        if item_tokens is None:
            continue
        text = torch.zeros(text_len, dtype=torch.float32, device=device)
        for i in items:
            ids = [t for t in item_tokens[i] if t < text_len]
            if ids:
                text[ids] = 1.0
        for r, cond in enumerate(rows_cond):
            if cond:
                out[s, r, image:] = text
    return out


def region_masks(spec_items, boxes, height: int, width: int, feather: int = 2, device=None) -> torch.Tensor:
    """(specs, height, width): where each LoRA's own pass is taken in the one-pass-per-LoRA mode, the union of its
    items' boxes with edges softened over `feather` latent cells; where boxes of different LoRAs overlap, they share."""
    out = torch.zeros(len(spec_items), 1, height, width, dtype=torch.float32, device=device)
    for s, items in enumerate(spec_items):
        for i in items:
            out[s, 0] = torch.maximum(out[s, 0], box_mask(boxes[i], height, width, device).view(height, width))
    if feather > 0:
        out = torch.nn.functional.avg_pool2d(out, 2 * feather + 1, stride=1, padding=feather, count_include_pad=False)
    out = out[:, 0]
    total = out.sum(0, keepdim=True)
    return torch.where(total > 1, out / total.clamp(min=1e-6), out)


def composite(base: torch.Tensor, passes, masks: torch.Tensor) -> torch.Tensor:
    """base + sum_s mask_s * (pass_s - base): each LoRA's pass inside its region, the pass without LoRAs elsewhere.
    base and passes (B, C, H, W), masks (S, H, W)."""
    out = base.clone()
    for mask, result in zip(masks, passes):
        out += mask.to(base.dtype)[None, None] * (result - base)
    return out


# ---- the low-rank arithmetic ----


@dataclass
class LowRank:
    up: torch.Tensor  # (out, rank)
    down: torch.Tensor  # (rank, in)
    scale: float  # strength * alpha / rank

    def to(self, device, dtype) -> "LowRank":
        return LowRank(self.up.to(device=device, dtype=dtype), self.down.to(device=device, dtype=dtype), self.scale)


def extract_lowrank(weights, strength: float = 1.0) -> LowRank | None:
    """weights = (up, down, alpha, mid, dora_scale, reshape), as Forge's LoRAAdapter keeps them; None for anything but
    a plain two-dimensional LoRA (LoCon mid weights, DoRA, reshapes stay global)."""
    up, down, alpha, mid, dora_scale, reshape = weights
    if mid is not None or dora_scale is not None or reshape is not None:
        return None
    if up.dim() != 2 or down.dim() != 2:
        return None
    rank = down.shape[0]
    a = float(alpha) / rank if alpha is not None else 1.0
    return LowRank(up, down, float(strength) * a)


def stack(parts, out_dim: int) -> LowRank:
    """Low-rank pieces that each land on a slice of a fused weight's rows ((start, size), LowRank) as one LowRank: a
    q, k or v LoRA on MOSS-Image's fused qkv projection."""
    rank = sum(lr.down.shape[0] for _, lr in parts)
    up = torch.zeros(out_dim, rank, dtype=parts[0][1].up.dtype)
    r = 0
    for (start, size), lr in parts:
        k = lr.down.shape[0]
        up[start:start + size, r:r + k] = lr.up * lr.scale
        r += k
    return LowRank(up, torch.cat([lr.down for _, lr in parts], dim=0), 1.0)


def combine(parts, out_dim: int) -> LowRank:
    """Every low-rank piece a LoRA file puts on one weight, as one LowRank. parts: [(offset, LowRank)], offset None for
    a piece on the whole weight, else Forge's (dim, start, size) slice. A file can carry both for MOSS-Image's fused
    qkv: proj.qkv on all of it and proj.q, proj.k, proj.v on its thirds."""
    if len(parts) == 1 and parts[0][0] is None:
        return parts[0][1]
    return stack([((0, out_dim) if offset is None else (offset[1], offset[2]), lr) for offset, lr in parts], out_dim)


def delta(x: torch.Tensor, lr: LowRank) -> torch.Tensor:
    return ((x @ lr.down.t()) @ lr.up.t()) * lr.scale


def subtract_outside(out: torch.Tensor, x: torch.Tensor, pairs) -> torch.Tensor:
    """out - sum_r (1 - w_r) * delta_r(x); pairs [(LowRank, w)] with w of shape (B, L) for x of shape (B, L, in).

    The (1 - w) factor and the scale go into the small (..., rank) product, and the sum lands in out in place when out
    is contiguous (no second activation-sized tensor)."""
    flat = out.reshape(-1, out.shape[-1])
    in_place = flat.data_ptr() == out.data_ptr() and out.is_contiguous()
    for lr, w in pairs:
        keep = ((1.0 - w) * lr.scale).to(x.dtype)
        h = ((x @ lr.down.t()) * keep.unsqueeze(-1)).to(out.dtype)
        up = lr.up.t().to(out.dtype)
        if in_place:
            flat.addmm_(h.reshape(-1, h.shape[-1]), up, alpha=-1.0)
        else:
            out = out - h @ up
    return out
