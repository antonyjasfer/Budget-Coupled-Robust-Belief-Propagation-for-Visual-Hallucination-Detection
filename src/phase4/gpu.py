"""Phase 4 GPU stages (PREREG v2). Lazy torch imports; every stage appends JSONL
rows atomically (one row per task) and resumes.

* run_vlm_captions: Qwen2.5-VL-3B-Instruct and InternVL3-2B (HF-native) captions
  through AutoModelForImageTextToText, greedy, same prompt as LLaVA.
* run_owl_tasks: OWLv2 scores for a list of (set, image, condition, categories)
  tasks; the condition is "clean", an S1 condition (e.g. "jpeg_3") or "s5" (the
  task's boxes are masked first).
LLaVA captions reuse src.phase1.gpu_pipeline.run_captions; the detector swaps
reuse src.phase3.gpu_extra.run_swap_detectors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Sequence

from PIL import Image

from src.phase1.gpu_pipeline import CAPTION_PROMPT, OWL_MODEL, QUERY_TEMPLATE, append_jsonl, read_done
from src.phase3.gpu_extra import cap_pixels
from src.phase4.shifts import corrupt, mask_boxes

VLMS = {
    "qwen": "Qwen/Qwen2.5-VL-3B-Instruct",
    "internvl": "OpenGVLab/InternVL3-2B-hf",
}
VLM_MAX_NEW_TOKENS = 512
VLM_MAX_PIXELS = 768 * 28 * 28


def shard(items: Sequence, k: int, n: int) -> list:
    """Every n-th item starting at k (stable across runs)."""
    return list(items)[k::n] if n > 1 else list(items)


def task_key(r: dict) -> tuple:
    return (r["set"], int(r["image_id"]), r["cond"])


def _commit(model):
    return getattr(getattr(model, "config", None), "_commit_hash", None)


def run_vlm_captions(name: str, image_paths: Dict[int, Path], out_path: Path,
                     log: Callable[[str], None] = print) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = VLMS[name]
    done = read_done(out_path, lambda r: int(r["image_id"]))
    todo = [i for i in image_paths if i not in done]
    log(f"{name} captions: {len(done)} done, {len(todo)} to go")
    if not todo:
        return
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.float16, device_map="auto").eval()
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto").eval()
    proc = AutoProcessor.from_pretrained(model_id)

    def build_inputs(img):
        # Official model-card pattern (image passed inside the chat template); fall back
        # to "template text + processor(images=...)" for processor versions without it.
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": CAPTION_PROMPT}]}]
        try:
            inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                              return_dict=True, return_tensors="pt")
        except Exception:
            msgs[0]["content"][0] = {"type": "image"}
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], images=[img], return_tensors="pt")
        inputs = inputs.to(model.device)
        for k in ("pixel_values",):
            if k in inputs and inputs[k].is_floating_point():
                inputs[k] = inputs[k].to(torch.float16)   # T4: fp16, not bf16
        return inputs

    for n, iid in enumerate(todo, 1):
        img = Image.open(image_paths[iid]).convert("RGB")
        if name == "qwen":
            img = cap_pixels(img, VLM_MAX_PIXELS)
        inputs = build_inputs(img)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=VLM_MAX_NEW_TOKENS, do_sample=False)
        gen = out[0, inputs["input_ids"].shape[1]:]
        caption = proc.batch_decode([gen], skip_special_tokens=True)[0].strip()
        append_jsonl(out_path, {
            "image_id": iid, "caption": caption, "model": model_id, "revision": _commit(model),
            "prompt": CAPTION_PROMPT, "max_new_tokens": VLM_MAX_NEW_TOKENS, "decoding": "greedy",
            "hit_max_tokens": int(gen.shape[0]) >= VLM_MAX_NEW_TOKENS - 1,
        })
        if n % 10 == 0 or n == len(todo):
            log(f"  {name} captions {n}/{len(todo)}")
    del model
    torch.cuda.empty_cache()


def run_owl_tasks(tasks: List[dict], out_path: Path, log: Callable[[str], None] = print) -> None:
    """tasks: {"set", "image_id", "cond", "path", "cats", optional "boxes"}."""
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    done = read_done(out_path, task_key)
    todo = [t for t in tasks if task_key(t) not in done and t["cats"]]
    log(f"owl: {len(done)} tasks done, {len(todo)} to go")
    if not todo:
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Owlv2Processor.from_pretrained(OWL_MODEL)
    owl = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL).to(dev).eval()
    rev = _commit(owl)
    cache: Dict[int, Image.Image] = {}
    for n, t in enumerate(todo, 1):
        iid = int(t["image_id"])
        base = cache.get(iid)
        if base is None:
            cache.clear()  # tasks are ordered by image, so one cached image suffices
            base = cache[iid] = Image.open(t["path"]).convert("RGB")
        if t["cond"] == "s5":
            img = mask_boxes(base, t["boxes"])
        else:
            img = corrupt(base, t["cond"], iid)
        with torch.inference_mode():
            enc = proc(text=[[QUERY_TEMPLATE.format(c) for c in t["cats"]]], images=img, return_tensors="pt").to(dev)
            scores = torch.sigmoid(owl(**enc).logits[0].float()).max(dim=0).values.tolist()
        append_jsonl(out_path, {"set": t["set"], "image_id": iid, "cond": t["cond"],
                                "scores": dict(zip(t["cats"], map(float, scores))),
                                "model": OWL_MODEL, "revision": rev})
        if n % 200 == 0 or n == len(todo):
            log(f"  owl {n}/{len(todo)}")
    del owl
    if dev == "cuda":
        torch.cuda.empty_cache()
