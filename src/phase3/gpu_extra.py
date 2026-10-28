"""Phase 3b GPU stages.

* run_swap_detectors: evidence for the SAME claims from two other open-vocabulary
  detectors (detector-swap shift), on clean images:
    - OWL-ViT v1 (google/owlvit-base-patch32): an older model of the same family;
    - Grounding DINO (IDEA-Research/grounding-dino-base): a different architecture
      (Swin + BERT, no CLIP backbone).
  For each claim we store the detector's maximum confidence that the object is
  present, as for OWLv2 in Phase 1.
* run_qwen_captions: detailed captions from a second VLM (Qwen2.5-VL-3B-Instruct)
  with the same prompt, greedy decoding (VLM-swap shift).

torch / transformers are imported lazily, as in src/phase1/gpu_pipeline.py, and
every stage appends to JSONL and resumes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Sequence

from PIL import Image

from src.phase1.gpu_pipeline import CAPTION_PROMPT, QUERY_TEMPLATE, append_jsonl, read_done

OWLV1_MODEL = "google/owlvit-base-patch32"
GDINO_MODEL = "IDEA-Research/grounding-dino-base"
QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_MAX_NEW_TOKENS = 384
QWEN_MAX_PIXELS = 768 * 28 * 28  # caps visual tokens so a T4 does not run out of memory


def gdino_phrase_positions(token_ids: Sequence[int], skip_ids: Sequence[int]) -> List[int]:
    """Positions of the tokens that belong to the object phrase in a Grounding DINO
    prompt such as "dog." ([CLS] dog . [SEP]): every token except special tokens
    and the final period."""
    skip = set(int(t) for t in skip_ids)
    return [k for k, t in enumerate(token_ids) if int(t) not in skip]


def cap_pixels(img: Image.Image, max_pixels: int) -> Image.Image:
    """Downscale (keeping aspect ratio) so that width * height <= max_pixels.
    Done here instead of via processor kwargs, whose names differ across
    transformers versions."""
    w, h = img.size
    if w * h <= max_pixels:
        return img
    s = (max_pixels / (w * h)) ** 0.5
    return img.resize((max(28, int(w * s)), max(28, int(h * s))), Image.BICUBIC)


def _commit(model):
    return getattr(getattr(model, "config", None), "_commit_hash", None)


def run_swap_detectors(image_paths: Dict[int, Path], claims_by_image: Dict[int, List[str]],
                       out_path: Path, log: Callable[[str], None] = print) -> None:
    import torch
    from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                              OwlViTForObjectDetection, OwlViTProcessor)

    done = read_done(out_path, lambda r: int(r["image_id"]))
    todo = [i for i in image_paths if claims_by_image.get(i) and i not in done]
    log(f"swap detectors: {len(done)} images done, {len(todo)} to go")
    if not todo:
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    v1_proc = OwlViTProcessor.from_pretrained(OWLV1_MODEL)
    v1 = OwlViTForObjectDetection.from_pretrained(OWLV1_MODEL).to(dev).eval()
    gd_proc = AutoProcessor.from_pretrained(GDINO_MODEL)
    gd = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL).to(dev).eval()
    tok = gd_proc.tokenizer
    skip = [tok.cls_token_id, tok.sep_token_id, tok.pad_token_id] + tok.convert_tokens_to_ids(["."])
    revs = {"owlvit_v1": _commit(v1), "grounding_dino": _commit(gd)}

    for n, iid in enumerate(todo, 1):
        img = Image.open(image_paths[iid]).convert("RGB")
        cats = claims_by_image[iid]
        with torch.inference_mode():
            enc = v1_proc(text=[[QUERY_TEMPLATE.format(c) for c in cats]], images=img, return_tensors="pt").to(dev)
            v1_scores = torch.sigmoid(v1(**enc).logits[0].float()).max(dim=0).values.tolist()
            gd_scores = []
            for c in cats:
                g = gd_proc(images=img, text=f"{c}.", return_tensors="pt").to(dev)
                logits = gd(**g).logits[0].float()                 # (queries, max_text_len)
                pos = gdino_phrase_positions(g["input_ids"][0].tolist(), skip)
                pos = [p for p in pos if p < logits.shape[1]]
                gd_scores.append(float(torch.sigmoid(logits[:, pos]).max()) if pos else float("nan"))
        append_jsonl(out_path, {
            "image_id": iid, "shift": "clean",
            "claims": [{"category": c, "owlvit_v1": float(s1), "grounding_dino": float(s2)}
                       for c, s1, s2 in zip(cats, v1_scores, gd_scores)],
            "models": {"owlvit_v1": OWLV1_MODEL, "grounding_dino": GDINO_MODEL}, "revisions": revs,
        })
        if n % 20 == 0 or n == len(todo):
            log(f"  swap detectors {n}/{len(todo)}")
    del v1, gd
    if dev == "cuda":
        torch.cuda.empty_cache()


def run_qwen_captions(image_paths: Dict[int, Path], out_path: Path, four_bit: bool = False,
                      log: Callable[[str], None] = print) -> None:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    done = read_done(out_path, lambda r: int(r["image_id"]))
    todo = [i for i in image_paths if i not in done]
    log(f"qwen captions: {len(done)} done, {len(todo)} to go")
    if not todo:
        return
    kw = {"device_map": "auto"}
    if four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    else:
        kw["dtype"] = torch.float16  # transformers >= 4.56 / 5.x name
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_MODEL, **kw).eval()
    except TypeError:  # older transformers: torch_dtype
        kw["torch_dtype"] = kw.pop("dtype", torch.float16)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_MODEL, **kw).eval()
    proc = AutoProcessor.from_pretrained(QWEN_MODEL)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": CAPTION_PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    for n, iid in enumerate(todo, 1):
        img = cap_pixels(Image.open(image_paths[iid]).convert("RGB"), QWEN_MAX_PIXELS)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=QWEN_MAX_NEW_TOKENS, do_sample=False)
        gen = out[0, inputs["input_ids"].shape[1]:]
        caption = proc.batch_decode([gen], skip_special_tokens=True)[0].strip()
        append_jsonl(out_path, {
            "image_id": iid, "caption": caption, "model": QWEN_MODEL, "revision": _commit(model),
            "prompt": CAPTION_PROMPT, "max_new_tokens": QWEN_MAX_NEW_TOKENS, "decoding": "greedy",
            "four_bit": four_bit, "hit_max_tokens": int(gen.shape[0]) >= QWEN_MAX_NEW_TOKENS - 1,
        })
        if n % 10 == 0 or n == len(todo):
            log(f"  qwen captions {n}/{len(todo)}")
    del model
    torch.cuda.empty_cache()
