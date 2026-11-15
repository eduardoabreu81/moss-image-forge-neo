# MOSS-Image support for Forge Neo
# Regional LoRA (regional.py, regional_lora.py): the layout's items, LoRA tags and boxes; the tokens of each item; the
# subtraction; and, on a small DiT, that the wrapped model equals the LoRA merged where the weights are 1 and the plain
# model where they are 0.

import copy
import importlib
import unittest
import unittest.mock

import torch

from forge_moss.native import regional, regional_lora
from forge_moss.native.layout import restore_layout_json
from forge_moss.native.transformer import MossImageDiT
from tests import forge_stubs
from tests.test_transformer import TINY, inputs, randomized

# a layout as Regional Boxes writes it, a LoRA in the first item's caption and one after the layout
LAYOUT = ('{"summary":{"canvas":"single","caption":"A snowy forest"},"items":['
          '{"label":"object","caption":"a red fox <lora:fox_v2:0.8> foxstyle","bbox":[-371,-395,270,139]},'
          '{"label":"object","caption":"a pine tree","bbox":[-512,200,511,511]}],'
          '"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"} <lora:turbo:1>')


class CharTokenizer:
    """One token per character: offsets are exact, as a fast tokenizer's."""

    is_fast = True

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        out = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


class FakeEngine:
    """The extension's text engine as far as regional_lora reads it: its tokenizer and token_ids."""

    def __init__(self, text_engine):
        self.te = text_engine
        self.tokenizer = type("T", (), {"tokenizer": CharTokenizer()})()

    def token_ids(self, text):
        full = self.te.TEMPLATE.format(restore_layout_json(text.strip()))
        return self.tokenizer.tokenizer(full)["input_ids"][:self.te.MAX_LENGTH]


class FakeNativeModel:
    """Forge's built-in MOSSImage engine, as far as regional_lora reads it."""

    def __init__(self):
        tokenizer = type("T", (), {"tokenizer": CharTokenizer()})()
        self.text_processing_engine_qwen = type("E", (), {"tokenizer": tokenizer, "llama_template": TEMPLATE_NATIVE})()


TEMPLATE_NATIVE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


class FakeModel:
    is_moss_image = True

    def __init__(self, text_engine):
        self.text_engine_moss = FakeEngine(text_engine)


class TestTags(unittest.TestCase):
    def test_mentions_follow_forge(self):
        self.assertEqual(regional.mentions("<lora:a:0.5> <lora:b> <lora:c:1:0.3> <lora:d:0.4:unet=0.9> <lora:e:x>"),
                         [regional.Mention("a", 0.5), regional.Mention("b", 1.0), regional.Mention("c", 0.3),
                          regional.Mention("d", 0.9)])
        self.assertEqual(regional.mentions("<lora:a:@1>"), [regional.Mention("a", 0.0)])
        self.assertEqual(regional.mentions("<hypernet:a:1> words"), [])
        self.assertEqual(regional.strip_networks("a <lora:x:1> b"), "a  b")

    def test_items_and_global_loras(self):
        items = regional.parse_items(LAYOUT)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].loras, [regional.Mention("fox_v2", 0.8)])
        self.assertEqual(items[1].loras, [])
        self.assertEqual(items[0].bbox, (-371.0, -395.0, 270.0, 139.0))
        self.assertTrue(LAYOUT[items[0].start:items[0].end].startswith('{"label"'))
        self.assertTrue(LAYOUT[items[0].start:items[0].end].endswith("]}"))
        self.assertEqual(regional.global_names(LAYOUT, items), {"turbo"})
        for text in ("A cat on a sofa.", '{"items": []}', '[{"bbox": [0, 0, 1, 1]}]', "", '{"summary": '):
            self.assertIsNone(regional.parse_items(text), text)
        # Forge's \( escape in a caption is no JSON escape, and still reads
        escaped = LAYOUT.replace("a pine tree", "a pine tree \\(tall\\)")
        self.assertEqual(len(regional.parse_items(escaped)), 2)

    def test_box_shares_match_regional_boxes(self):
        top, left, bottom, right = regional.box_fraction((-371, -395, 270, 139), 1024, 1024)
        self.assertAlmostEqual(top, 0.1376953125)
        self.assertAlmostEqual(left, 0.1142578125)
        self.assertAlmostEqual(bottom, 0.763671875)
        self.assertAlmostEqual(right, 0.6357421875)
        self.assertEqual(regional.box_fraction((-624, -416, 624, 416), 832, 1248), (0.0, 0.0, 1.0, 1.0))
        self.assertEqual(regional.box_fraction((-900, -900, 900, 900), 832, 1248), (0.0, 0.0, 1.0, 1.0))

    def test_unescape_map(self):
        text, index = regional.unescape_with_map("a \\(b\\) c")
        self.assertEqual(text, "a (b) c")
        self.assertEqual(index[2], 2)
        self.assertEqual(index[4], 3)
        self.assertEqual(index[len("a \\(b\\) c")], len(text))

    def test_insertion_map(self):
        original = '"a":1,"b":["c":2]'
        rebuilt = '{"a":1,"b":[{"c":2}]}'
        mapping = regional.insertion_map(original, rebuilt)
        self.assertEqual(len(mapping), len(rebuilt) + 1)
        self.assertEqual(mapping[0], 0)  # the inserted { stands for where the text goes on
        self.assertEqual(original[mapping[rebuilt.index('"c"')]], '"')
        self.assertEqual(mapping[rebuilt.index('"c"')], original.index('"c"'))
        self.assertEqual(mapping[len(rebuilt)], len(original))
        # past a difference that is no inserted brace, everything goes to the end
        self.assertEqual(regional.insertion_map("ab, x", "ab x")[-2], 5)

    def test_token_spans(self):
        offsets = [(0, 2), (2, 4), (4, 6), (6, 7), (0, 0)]
        self.assertEqual(regional.token_spans(offsets, [(0, 4), (3, 7)]), [[0, 1], [2, 3]])


class TestWeights(unittest.TestCase):
    def test_box_mask(self):
        mask = regional.box_mask((0.0, 0.5, 0.5, 1.0), 4, 4).view(4, 4)
        self.assertEqual(mask.sum().item(), 4)
        self.assertEqual(mask[0, 2].item(), 1)
        self.assertEqual(mask[3, 3].item(), 0)
        # a box thinner than a cell still covers one
        self.assertEqual(regional.box_mask((0.99, 0.99, 1.0, 1.0), 4, 4).sum().item(), 1)

    def test_token_weights(self):
        boxes = [(0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0)]
        weights = regional.token_weights([[0], [0, 1]], boxes, [[0, 1], [3]], [True, False], 2, 2, 5)
        self.assertEqual(weights.shape, (2, 2, 9))
        self.assertEqual(weights[0, 0, :4].tolist(), [1, 0, 0, 0])
        self.assertEqual(weights[1, 0, :4].tolist(), [1, 0, 0, 1])
        self.assertEqual(weights[0, 0, 4:].tolist(), [1, 1, 0, 0, 0])
        self.assertEqual(weights[1, 0, 4:].tolist(), [1, 1, 0, 1, 0])
        # the negative prompt's row keeps the boxes but not the items' text
        self.assertEqual(weights[1, 1, :4].tolist(), [1, 0, 0, 1])
        self.assertEqual(weights[1, 1, 4:].tolist(), [0, 0, 0, 0, 0])
        # the whole prompt: every text token of the positive pass, none of the negative's; the picture still boxed
        whole = regional.token_weights([[0], []], boxes, [[0, 1], [3]], [True, False], 2, 2, 5, all_text=True)
        self.assertEqual(whole[0, 0].tolist(), [1, 0, 0, 0, 1, 1, 1, 1, 1])
        self.assertEqual(whole[0, 1].tolist(), [1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(whole[1].sum().item(), 0)
        image_only = regional.token_weights([[1]], boxes, None, [True], 2, 2, 3)
        self.assertEqual(image_only[0, 0].tolist(), [0, 0, 0, 1, 0, 0, 0])


class TestPerLoraMasks(unittest.TestCase):
    def test_region_masks(self):
        # the second box touches the first one's softened edge, not its corner
        masks = regional.region_masks([[0], [1]], [(0.0, 0.0, 0.5, 0.5), (0.5, 0.25, 1.0, 0.75)], 8, 8)
        self.assertEqual(masks.shape, (2, 8, 8))
        self.assertEqual(masks[0, 0, 0].item(), 1.0)  # well inside its box
        self.assertEqual(masks[0, 7, 7].item(), 0.0)  # far from it
        self.assertTrue(0.0 < masks[0, 4, 1].item() < 1.0)  # softened past its edge
        self.assertTrue(bool((masks.sum(0) <= 1.0 + 1e-6).all()))  # overlapping boxes share
        hard = regional.region_masks([[0]], [(0.0, 0.0, 0.5, 0.5)], 4, 4, feather=0)
        self.assertEqual(hard[0].tolist(), [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    def test_composite(self):
        base = torch.zeros(1, 2, 2, 2)
        one, two = torch.ones(1, 2, 2, 2), torch.full((1, 2, 2, 2), 2.0)
        masks = torch.tensor([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.5]]])
        out = regional.composite(base, [one, two], masks)
        self.assertEqual(out[0, 0].tolist(), [[1.0, 0.0], [0.0, 1.0]])


class TestArithmetic(unittest.TestCase):
    def test_subtract_outside(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(2, 5, 6, generator=g)
        base = torch.randn(4, 6, generator=g)
        lr = regional.LowRank(torch.randn(4, 3, generator=g), torch.randn(3, 6, generator=g), 0.7)
        full = x @ base.t() + regional.delta(x, lr)
        w = torch.tensor([[1, 0, 1, 0, 0], [0, 0, 0, 1, 1]], dtype=torch.float32)
        out = regional.subtract_outside(full.clone(), x, [(lr, w)])
        expected = x @ base.t() + w.unsqueeze(-1) * regional.delta(x, lr)
        torch.testing.assert_close(out, expected)
        # a non-contiguous output takes the same values out of place
        strided = full.clone().transpose(0, 1).contiguous().transpose(0, 1)
        torch.testing.assert_close(regional.subtract_outside(strided, x, [(lr, w)]), expected)

    def test_extract_and_stack(self):
        up, down = torch.ones(8, 2), torch.ones(2, 4)
        self.assertAlmostEqual(regional.extract_lowrank((up, down, 4.0, None, None, None), 0.5).scale, 1.0)
        self.assertIsNone(regional.extract_lowrank((up, down, None, torch.ones(2, 2), None, None)))
        q = regional.LowRank(torch.full((3, 1), 2.0), torch.ones(1, 4), 0.5)
        v = regional.LowRank(torch.full((3, 2), 1.0), torch.ones(2, 4), 1.0)
        fused = regional.stack([((0, 3), q), ((6, 3), v)], 9)
        merged = fused.up @ fused.down * fused.scale
        torch.testing.assert_close(merged[:3], q.up @ q.down * q.scale)
        torch.testing.assert_close(merged[3:6], torch.zeros(3, 4))
        torch.testing.assert_close(merged[6:], v.up @ v.down * v.scale)

    def test_combine_whole_and_sliced_pieces(self):
        # a LoRA with proj.qkv and proj.q on a fused qkv weight: both land on it, added up
        g = torch.Generator().manual_seed(1)
        whole = regional.LowRank(torch.randn(9, 2, generator=g), torch.randn(2, 4, generator=g), 0.5)
        q = regional.LowRank(torch.randn(3, 1, generator=g), torch.randn(1, 4, generator=g), 2.0)
        one = regional.combine([(None, whole), ((0, 3, 3), q)], 9)
        expected = whole.up @ whole.down * whole.scale
        expected[3:6] += q.up @ q.down * q.scale
        torch.testing.assert_close(one.up @ one.down * one.scale, expected)
        self.assertIs(regional.combine([(None, whole)], 9), whole)


class TestCollect(unittest.TestCase):
    # text_engine imports Forge: regional_lora reads it under the same stubs as tests/test_text_engine.py
    def setUp(self):
        self.ctx = forge_stubs.installed(forge_stubs.text_processing())
        self.ctx.__enter__()
        self.te = importlib.import_module("forge_moss.native.text_engine")
        self.rl = importlib.import_module("forge_moss.native.regional_lora")
        self.model = FakeModel(self.te)

    def tearDown(self):
        self.ctx.__exit__(None, None, None)

    def test_items_tokens_and_boxes(self):
        TEMPLATE = self.te.TEMPLATE
        layouts, global_names = self.rl.collect(self.model, self.rl.EXTENSION, [LAYOUT, LAYOUT], 1024, 1024)
        self.assertEqual(global_names, {"turbo"})
        self.assertEqual(len(layouts), 1)
        count, layout = next(iter(layouts.items()))
        body = restore_layout_json(regional.strip_networks(LAYOUT).strip())
        self.assertEqual(count, len(TEMPLATE.format(body)))
        prefix = TEMPLATE.index("{}")
        first = "".join(chr(ord(TEMPLATE.format(body)[t])) for t in layout.item_tokens[0])
        self.assertEqual(first, body[body.index('{"label"'):body.index('},{"label"') + 1])
        self.assertEqual(layout.item_tokens[0][0], prefix + body.index('{"label"'))
        self.assertEqual([m.name for m in layout.loras[0]], ["fox_v2"])
        self.assertAlmostEqual(layout.boxes[1][1], (200 + 512) / 1024)

    def test_brace_stripped_layout(self):
        stripped = LAYOUT.replace("{", "").replace("}", "")
        layouts, _ = self.rl.collect(self.model, self.rl.EXTENSION, [stripped], 1024, 1024)
        self.assertEqual(len(layouts), 1)
        self.assertEqual([m.name for m in next(iter(layouts.values())).loras[0]], ["fox_v2"])

    def test_forge_built_in_backend(self):
        # Forge's engine reads the prompt as it is: a stripped layout is located through the rebuilt one, and Forge's
        # \( escape is one character to its tokenizer
        model = FakeNativeModel()
        # (a stripped layout with \( escapes is not rebuilt: layout.restore_layout_json needs it to parse as JSON)
        escaped = LAYOUT.replace("a pine tree", "a pine tree \\(tall\\)")
        plain = LAYOUT.replace("a pine tree", "a pine tree (tall)")
        for prompt in (escaped, plain.replace("{", "").replace("}", "")):
            layouts, global_names = self.rl.collect(model, self.rl.NATIVE, [prompt], 1024, 1024)
            self.assertEqual(global_names, {"turbo"}, prompt)
            count, layout = next(iter(layouts.items()))
            body = regional.strip_networks(prompt).strip()
            text = regional.unescape_with_map(TEMPLATE_NATIVE.format(body))[0]
            self.assertEqual(count, len(text))
            first = "".join(text[t] for t in layout.item_tokens[0])
            second = "".join(text[t] for t in layout.item_tokens[1])
            self.assertTrue(first.lstrip("{").startswith('"label":"object","caption":"a red fox'), first)
            self.assertTrue(first.rstrip("}").endswith("[-371,-395,270,139]"), first)
            self.assertIn("a pine tree (tall)", second)
            self.assertEqual([m.name for m in layout.loras[0]], ["fox_v2"])

    def test_no_layout_or_no_regional_lora(self):
        for prompt in ("A cat <lora:fox_v2:1>", LAYOUT.replace("<lora:fox_v2:0.8> ", "")):
            layouts, _ = self.rl.collect(self.model, self.rl.EXTENSION, [prompt], 1024, 1024)
            self.assertEqual(layouts, {}, prompt)


def tiny_dit(seed=0):
    return randomized(MossImageDiT(**TINY, fused_qkv=True), seed).eval()


def lowranks(seed=3):
    g = torch.Generator().manual_seed(seed)
    dim, ffn, text = TINY["dim"], TINY["ffn_dim"], TINY["text_dim"]

    def pair(out, inp, rank=2):
        return regional.LowRank(torch.randn(out, rank, generator=g) * 0.2, torch.randn(rank, inp, generator=g) * 0.2, 0.9)

    return {
        "layers.0.proj.o": pair(dim, dim),
        "layers.1.proj.qkv": regional.stack([((0, dim), pair(dim, dim))], 3 * dim),  # a q LoRA on the fused qkv
        "layers.2.proj.ffn_down": pair(dim, ffn),
        "text_embedder.proj": pair(dim, text),
        "text_embedder.blocks.0.ffn_up": pair(ffn, dim),
    }


def merged(model, layers):
    full = copy.deepcopy(model)
    with torch.no_grad():
        for path, lr in layers.items():
            full.get_submodule(path).weight += lr.up @ lr.down * lr.scale
    return full


class TestPatchedModel(unittest.TestCase):
    def run_patched(self, model, layers, items, layout, x, sigma, context, options=None):
        spec = regional_lora.Spec("fox", items, 1.0, layers)
        regional_lora.patch(model, [spec], {context.shape[1]: layout}, regional_lora.EXTENSION)
        try:
            with torch.no_grad():
                return model(x, sigma, context=context, transformer_options=options)
        finally:
            regional_lora.release()

    def test_weights_one_is_the_lora_and_zero_is_none(self):
        # Forge hands over the model with the LoRA merged; the regional LoRA takes it away where the weights are 0
        model, layers = tiny_dit(), lowranks()
        lora = merged(model, layers)
        x, sigma, context = inputs(batch=2, height=3, width=5, text=7)
        with torch.no_grad():
            base = model(x, sigma, context)
            full = lora(x, sigma, context)
        everywhere = regional_lora.Layout([(0.0, 0.0, 1.0, 1.0)], [list(range(7))], [[]])
        out = self.run_patched(lora, layers, [0], everywhere, x, sigma, context)
        torch.testing.assert_close(out, full, rtol=1e-4, atol=1e-5)
        # a spec with no item keeps the LoRA nowhere
        out = self.run_patched(lora, layers, [], everywhere, x, sigma, context)
        torch.testing.assert_close(out, base, rtol=1e-4, atol=1e-5)
        # undone: the LoRA model is whole again, with no forward left on its instances
        self.assertNotIn("forward", lora.__dict__)
        self.assertFalse(any("forward" in m.__dict__ for m in lora.modules()))
        with torch.no_grad():
            torch.testing.assert_close(lora(x, sigma, context), full)

    def test_a_box_changes_only_part_of_the_picture_at_the_first_layer(self):
        # with the LoRA on the first block's output projection only, a box in the corner changes the result, and the
        # LoRA's text weights reach only the positive pass
        model = tiny_dit()
        layers = {"layers.0.proj.o": lowranks()["layers.0.proj.o"]}
        x, sigma, context = inputs(batch=2, height=4, width=4, text=6)
        lora = merged(model, layers)
        with torch.no_grad():
            base = model(x, sigma, context)
            full = lora(x, sigma, context)
        corner = regional_lora.Layout([(0.0, 0.0, 0.5, 0.5)], [[0, 1]], [[]])
        out = self.run_patched(lora, layers, [0], corner, x, sigma, context, {"cond_or_uncond": [0, 1]})
        self.assertFalse(torch.allclose(out, base))
        self.assertFalse(torch.allclose(out, full))

    def test_the_last_layer_changes_exactly_the_box(self):
        # the final layer reads each image token on its own: inside the box the picture is the LoRA's, outside it is
        # the plain model's, cell for cell
        model = tiny_dit()
        g = torch.Generator().manual_seed(5)
        layers = {"final_linear_video": regional.LowRank(torch.randn(TINY["latent_channels"], 2, generator=g),
                                                         torch.randn(2, TINY["dim"], generator=g), 0.5)}
        lora = merged(model, layers)
        x, sigma, context = inputs(batch=2, height=4, width=4, text=6)
        with torch.no_grad():
            base = model(x, sigma, context)
            full = lora(x, sigma, context)
        corner = regional_lora.Layout([(0.0, 0.0, 0.5, 0.5)], [[0, 1]], [[]])
        out = self.run_patched(lora, layers, [0], corner, x, sigma, context)
        inside = torch.zeros(4, 4, dtype=torch.bool)
        inside[:2, :2] = True
        torch.testing.assert_close(out[..., inside], full[..., inside], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(out[..., ~inside], base[..., ~inside], rtol=1e-4, atol=1e-5)

    def test_one_pass_per_lora(self):
        # each LoRA's own pass inside its box, the pass without LoRAs far from it, the LoRA's model where it covers all
        model, layers = tiny_dit(), lowranks()
        lora = merged(model, layers)
        x, sigma, context = inputs(batch=1, height=8, width=8, text=6)
        with torch.no_grad():
            base = model(x, sigma, context)
            full = lora(x, sigma, context)
        try:
            regional_lora.state.multipass = False
            for box, check in (((0.0, 0.0, 1.0, 1.0), "all"), ((0.0, 0.0, 0.5, 0.5), "corner")):
                spec = regional_lora.Spec("fox", [0], 1.0, layers)
                layout = regional_lora.Layout([box], [[0, 1]], [[]])
                regional_lora.patch(lora, [spec], {6: layout}, regional_lora.EXTENSION)
                regional_lora.state.multipass = True
                with torch.no_grad():
                    out = lora(x, sigma, context=context)
                regional_lora.release()
                if check == "all":
                    torch.testing.assert_close(out, full, rtol=1e-4, atol=1e-5)
                else:
                    torch.testing.assert_close(out[..., :2, :2], full[..., :2, :2], rtol=1e-4, atol=1e-5)
                    torch.testing.assert_close(out[..., 6:, 6:], base[..., 6:, 6:], rtol=1e-4, atol=1e-5)
        finally:
            regional_lora.release()
        self.assertFalse(regional_lora.state.multipass)

    def test_reference_images_keep_the_lora_whole(self):
        # MOSS Image Edit's references run through reference_kv: no box applies to them, even at the picture's size
        model = tiny_dit()
        layers = {"layers.0.proj.o": lowranks()["layers.0.proj.o"]}
        lora = merged(model, layers)
        calls = []
        weights_for = regional_lora._weights_for

        def recording(x):
            result = weights_for(x)
            calls.append((regional_lora.state.in_reference, result is not None))
            return result

        spec = regional_lora.Spec("fox", [0], 1.0, layers)
        x, sigma, context = inputs(batch=1, height=4, width=4, text=6)
        reference = torch.randn(1, TINY["latent_channels"], 4, 4)
        regional_lora.patch(lora, [spec], {6: regional_lora.Layout([(0.0, 0.0, 0.5, 0.5)], [[0]], [[]])},
                            regional_lora.EXTENSION)
        try:
            with unittest.mock.patch.object(regional_lora, "_weights_for", recording), torch.no_grad():
                lora(x, sigma, context=context, ref_latents=[reference])
        finally:
            regional_lora.release()
        # the reference pass reached the layer with no weights, the picture's pass with them
        self.assertIn((True, False), calls)
        self.assertIn((False, True), calls)
        self.assertNotIn((True, True), calls)
        self.assertNotIn("reference_kv", lora.__dict__)

if __name__ == "__main__":
    unittest.main()
