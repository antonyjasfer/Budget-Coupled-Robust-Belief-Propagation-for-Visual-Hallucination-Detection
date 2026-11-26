"""Phase 5 GPU stage: captions from six generators for the transfer matrix.

Every generator writes append-only JSONL (one row per image) and resumes from what is
already written, so a Kaggle session can stop at any point. Run one process per GPU
with a different shard. Not runnable in a CPU-only workspace; smoke-test it first with
`scripts/phase5.py captions --limit 20` on Kaggle.

T4 notes: no bfloat16, so models load in float16; `four_bit=True` loads in NF4 with
float16 compute for models that do not fit (LLaVA-OneVision-7B on one T4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

from PIL import Image

from src.phase1.gpu_pipeline import CAPTION_PROMPT, append_jsonl, read_done
from src.phase3.gpu_extra import cap_pixels

MAX_NEW_TOKENS = 512
MAX_PIXELS = 768 * 28 * 28


@dataclass(frozen=True)
class Generator:
    hf_id: str
    sample: bool = False            # greedy unless True
    temperature: float = 1.0
    top_p: float = 0.9
    cap_pixels: bool = False        # Qwen-family processors take any resolution
    four_bit_default: bool = False
    chat_template: bool = True      # False: plain prompt straight through the processor (InstructBLIP)
    extra: Dict = field(default_factory=dict)


# Model ids as in the Q1 plan (Part D.1); check each card before the pilot.
GENERATORS: Dict[str, Generator] = {
    "llava": Generator("llava-hf/llava-1.5-7b-hf"),
    "llava_t1": Generator("llava-hf/llava-1.5-7b-hf", sample=True),     # controlled prevalence knob
    "qwen25": Generator("Qwen/Qwen2.5-VL-3B-Instruct", cap_pixels=True),
    "qwen3_4b": Generator("Qwen/Qwen3-VL-4B-Instruct", cap_pixels=True),
    "internvl3": Generator("OpenGVLab/InternVL3-2B-hf"),
    # The chat-template route gives InstructBLIP a 5-D pixel_values (pilot, transformers 5.20): use the plain prompt.
    "instructblip": Generator("Salesforce/instructblip-vicuna-7b", four_bit_default=True, chat_template=False),
    "llava_ov": Generator("llava-hf/llava-onevision-qwen2-7b-ov-hf", four_bit_default=True),
}


def _load(g: Generator, four_bit: bool):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    kw = {"device_map": "auto"}
    if four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16)
    try:
        model = AutoModelForImageTextToText.from_pretrained(g.hf_id, dtype=torch.float16, **kw)
    except TypeError:  # older transformers
        model = AutoModelForImageTextToText.from_pretrained(g.hf_id, torch_dtype=torch.float16, **kw)
    return model.eval(), AutoProcessor.from_pretrained(g.hf_id)


def _inputs(proc, img, device, chat_template: bool = True):
    """Chat template with the image inside, else template text + images, else the raw
    prompt. chat_template=False goes straight to the raw prompt (InstructBLIP)."""
    import torch
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": CAPTION_PROMPT}]}]
    if not chat_template:
        x = proc(images=img, text=CAPTION_PROMPT, return_tensors="pt")
    else:
        try:
            x = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                         return_tensors="pt")
        except Exception:
            try:
                msgs[0]["content"][0] = {"type": "image"}
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = CAPTION_PROMPT
            x = proc(images=img, text=text, return_tensors="pt")
    x = x.to(device)
    for k in list(x.keys()):
        if k.startswith("pixel_values") and x[k].is_floating_point():
            x[k] = x[k].to(torch.float16)
    return x


def _run_llava_sampled(image_paths: Dict[int, Path], out_path: Path, log: Callable[[str], None]) -> None:
    """The source pipeline (4-bit LLaVA-1.5-7B at the pinned revision, same prompt) with
    nucleus sampling instead of greedy decoding: a controlled change of prevalence."""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
    from src.phase1.gpu_pipeline import LLAVA_MODEL, LLAVA_REVISION
    g = GENERATORS["llava_t1"]
    done = read_done(out_path, lambda r: int(r["image_id"]))
    todo = [i for i in image_paths if i not in done]
    log(f"llava_t1: {len(done)} done, {len(todo)} to go")
    if not todo:
        return
    proc = AutoProcessor.from_pretrained(LLAVA_MODEL, revision=LLAVA_REVISION)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_MODEL, revision=LLAVA_REVISION, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)).eval()
    prompt = f"USER: <image>\n{CAPTION_PROMPT} ASSISTANT:"
    for n, iid in enumerate(todo, 1):
        x = proc(images=Image.open(image_paths[iid]).convert("RGB"), text=prompt, return_tensors="pt").to(model.device)
        torch.manual_seed(int(iid))
        with torch.inference_mode():
            out = model.generate(**x, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                                 temperature=g.temperature, top_p=g.top_p)
        gen = out[0, x["input_ids"].shape[1]:]
        append_jsonl(out_path, {
            "image_id": iid, "caption": proc.tokenizer.decode(gen, skip_special_tokens=True).strip(),
            "generator": "llava_t1", "model": LLAVA_MODEL, "revision": LLAVA_REVISION, "prompt": CAPTION_PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS, "decoding": f"sample T={g.temperature} top_p={g.top_p} seed=image_id",
            "four_bit": True, "hit_max_tokens": int(gen.shape[0]) >= MAX_NEW_TOKENS - 1})
        if n % 25 == 0 or n == len(todo):
            log(f"  llava_t1 {n}/{len(todo)}")
    del model
    torch.cuda.empty_cache()


def run_generator(name: str, image_paths: Dict[int, Path], out_path: Path, four_bit: Optional[bool] = None,
                  log: Callable[[str], None] = print) -> None:
    """Generators used in Phase 4 run through the exact Phase 4 code, so their captions are
    comparable with the published results; the others use the generic loader below."""
    if name == "llava":
        from src.phase1.gpu_pipeline import run_captions
        return run_captions(image_paths, out_path, log=log)
    if name in ("qwen25", "internvl3"):
        from src.phase4.gpu import run_vlm_captions
        return run_vlm_captions({"qwen25": "qwen", "internvl3": "internvl"}[name], image_paths, out_path, log=log)
    if name == "llava_t1":
        return _run_llava_sampled(image_paths, out_path, log)
    if name == "instructblip":
        return _run_instructblip(image_paths, out_path, four_bit, log)
    import torch
    g = GENERATORS[name]
    done = read_done(out_path, lambda r: int(r["image_id"]))
    todo = [i for i in image_paths if i not in done]
    log(f"{name}: {len(done)} done, {len(todo)} to go")
    if not todo:
        return
    fb = g.four_bit_default if four_bit is None else four_bit
    model, proc = _load(g, fb)
    rev = getattr(getattr(model, "config", None), "_commit_hash", None)
    for n, iid in enumerate(todo, 1):
        img = Image.open(image_paths[iid]).convert("RGB")
        if g.cap_pixels:
            img = cap_pixels(img, MAX_PIXELS)
        x = _inputs(proc, img, model.device, chat_template=g.chat_template)
        gen_kw = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=g.sample)
        if g.sample:
            torch.manual_seed(int(iid))          # one seed per image: reruns give the same caption
            gen_kw.update(temperature=g.temperature, top_p=g.top_p)
        with torch.inference_mode():
            out = model.generate(**x, **gen_kw)[0]
        L = x["input_ids"].shape[1]
        if out.shape[0] >= L and torch.equal(out[:L].cpu(), x["input_ids"][0].cpu()):
            out = out[L:]                        # decoder-only models echo the prompt
        caption = proc.batch_decode([out], skip_special_tokens=True)[0].strip()
        append_jsonl(out_path, {
            "image_id": iid, "caption": caption, "generator": name, "model": g.hf_id, "revision": rev,
            "prompt": CAPTION_PROMPT, "max_new_tokens": MAX_NEW_TOKENS,
            "decoding": f"sample T={g.temperature} top_p={g.top_p} seed=image_id" if g.sample else "greedy",
            "four_bit": fb, "hit_max_tokens": int(out.shape[0]) >= MAX_NEW_TOKENS - 1,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        })
        if n % 25 == 0 or n == len(todo):
            log(f"  {name} {n}/{len(todo)}")
    del model
    torch.cuda.empty_cache()


def _run_instructblip(image_paths, out_path, four_bit, log):
    """InstructBLIP via its own classes on one GPU (shard k -> GPU k). The generic route
    (AutoProcessor, device_map='auto' across two T4s, transformers 5.20) gave the same
    non-text caption for every image; see prereg-v3, deviation D1."""
    import re, torch
    from transformers import BitsAndBytesConfig, InstructBlipForConditionalGeneration, InstructBlipProcessor
    g = GENERATORS["instructblip"]
    done = read_done(out_path, lambda r: int(r["image_id"]))
    todo = [i for i in image_paths if i not in done]
    log(f"instructblip: {len(done)} done, {len(todo)} to go")
    if not todo:
        return
    m = re.search(r"\.s(\d+)\.jsonl$", str(out_path))
    dev = (int(m.group(1)) if m else 0) % max(torch.cuda.device_count(), 1)
    fb = g.four_bit_default if four_bit is None else four_bit
    kw = dict(device_map={"": dev}, dtype=torch.float16)
    if fb:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16)
    model = InstructBlipForConditionalGeneration.from_pretrained(g.hf_id, **kw).eval()
    proc = InstructBlipProcessor.from_pretrained(g.hf_id)
    rev = getattr(model.config, "_commit_hash", None)
    vdt = next(model.vision_model.parameters()).dtype
    first = []
    for n, iid in enumerate(todo, 1):
        x = proc(images=Image.open(image_paths[iid]).convert("RGB"), text=CAPTION_PROMPT, return_tensors="pt")
        x = {k: v.to(dev) for k, v in x.items()}
        if x["pixel_values"].ndim == 5:
            x["pixel_values"] = x["pixel_values"].flatten(0, 1)
        x["pixel_values"] = x["pixel_values"].to(vdt)
        with torch.inference_mode():
            out = model.generate(**x, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)[0]
        L = x["input_ids"].shape[1]
        if out.shape[0] >= L and torch.equal(out[:L].cpu(), x["input_ids"][0].cpu()):
            out = out[L:]
        caption = proc.batch_decode([out], skip_special_tokens=True)[0].strip()
        if caption.startswith(CAPTION_PROMPT):
            caption = caption[len(CAPTION_PROMPT):].strip()
        if n <= 5:
            first.append(caption)
            log(f"  instructblip sample {n}: {caption[:120]}")
            if n == 5 and len(set(first)) == 1:
                raise RuntimeError("instructblip: first 5 captions identical, stopping")
        append_jsonl(out_path, {
            "image_id": iid, "caption": caption, "generator": "instructblip", "model": g.hf_id, "revision": rev,
            "prompt": CAPTION_PROMPT, "max_new_tokens": MAX_NEW_TOKENS, "decoding": "greedy",
            "four_bit": fb, "hit_max_tokens": int(out.shape[0]) >= MAX_NEW_TOKENS - 1,
            "gpu": torch.cuda.get_device_name(dev)})
        if n % 25 == 0 or n == len(todo):
            log(f"  instructblip {n}/{len(todo)}")
    del model
    torch.cuda.empty_cache()
