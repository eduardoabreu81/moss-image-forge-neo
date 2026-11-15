# MOSS-Image support for Forge Neo
# Regional LoRA: a <lora:...> tag inside one item of a layout prompt acts only inside that item's box. No UI of its own:
# Settings > Stable Diffusion > MOSS-Image turns it off. The work is in forge_moss/native/regional_lora.py; this script
# only calls it before each sampling and undoes it after the batch.

import sys
import traceback
from pathlib import Path

_root = str(Path(__file__).resolve().parents[1])
if _root not in sys.path:
    sys.path.insert(0, _root)

from modules import scripts


def _regional():
    from forge_moss.native import regional_lora

    return regional_lora


class MossRegionalLora(scripts.Script):
    def title(self):
        return "MOSS-Image Regional LoRA"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def process_before_every_sampling(self, p, *args, **kwargs):
        try:
            _regional().prepare(p)
        except Exception:
            print("[MOSS-Image] regional LoRA failed; the LoRAs stay global:")
            traceback.print_exc()

    def postprocess_batch(self, p, *args, **kwargs):
        self._release()

    def postprocess(self, p, processed, *args):
        self._release()

    @staticmethod
    def _release():
        try:
            _regional().release()
        except Exception:
            traceback.print_exc()
