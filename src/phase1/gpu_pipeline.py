"""Phase 1 GPU stages: LLaVA captions with token probabilities, OWLv2 + CLIP evidence.

torch / transformers are imported lazily so the pure helpers below can be
unit-tested on CPU without them. Every stage appends to a JSONL file and skips
items already present, so a disconnected Colab/Kaggle session can resume.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageFilter

LLAVA_MODEL = "llava-hf/llava-1.5-7b-hf"
LLAVA_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"  # pinned in M5
OWL_MODEL = "google/owlv2-base-patch16-ensemble"
CLIP_MODEL = "openai/clip-vit-large-patch14"
CAPTION_PROMPT = "Please describe this image in detail."
QUERY_TEMPLATE = "a photo of a {}"
MAX_NEW_TOKENS = 512

SHIFTS = ("clean", "blur", "jpeg", "downsample")


# ----------------------------------------------------------------------------
# Pure helpers (CPU, unit-tested)
# ----------------------------------------------------------------------------

def apply_shift(img: Image.Image, shift: str) -> Image.Image:
    """Deterministic evidence shifts used to measure delta_i."""
    img = img.convert("RGB")
    if shift == "clean":
        return img
    if shift == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=3))
    if shift == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=10)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if shift == "downsample":
        w, h = img.size
        small = img.resize((max(1, w // 4), max(1, h // 4)), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)
    raise ValueError(f"unknown shift {shift}")


def token_char_offsets(decode: Callable[[List[int]], str], ids: Sequence[int]) -> List[Tuple[int, int]]:
    """Character span of each generated token in the decoded text (cumulative decoding)."""
    spans, prev = [], 0
    for k in range(len(ids)):
        cur = len(decode(list(ids[: k + 1])))
        spans.append((prev, max(prev, cur)))
        prev = max(prev, cur)
    return spans


def span_token_logprobs(
    token_spans: Sequence[Tuple[int, int]],
    token_logprobs: Sequence[float],
    char_span: Tuple[int, int],
    text_offset: int = 0,
) -> List[float]:
    """Log-probs of the generated tokens overlapping a character span of the caption.

    `text_offset` = number of leading characters stripped from the decoded text
    to obtain the caption (char_span is relative to the caption).
    """
    a, b = char_span[0] + text_offset, char_span[1] + text_offset
    return [lp for (s, e), lp in zip(token_spans, token_logprobs) if e > a and s < b and e > s]


def claim_token_features(logprobs: Sequence[float]) -> Dict[str, Optional[float]]:
    if not logprobs:
        return {"tok_logprob_first": None, "tok_logprob_mean": None, "tok_prob_min": None}
    return {
        "tok_logprob_first": float(logprobs[0]),
        "tok_logprob_mean": float(sum(logprobs) / len(logprobs)),
        "tok_prob_min": float(min(math.exp(x) for x in logprobs)),
    }


def owl_box_to_pixels(box_cxcywh: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    """OWLv2 boxes are normalised to the zero-padded SQUARE input (side = max(w, h))."""
    side = max(width, height)
    cx, cy, bw, bh = (float(v) * side for v in box_cxcywh)
    x0 = int(max(0, math.floor(cx - bw / 2)))
    y0 = int(max(0, math.floor(cy - bh / 2)))
    x1 = int(min(width, math.ceil(cx + bw / 2)))
    y1 = int(min(height, math.ceil(cy + bh / 2)))
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)
    return x0, y0, x1, y1


def padded_crop(img: Image.Image, box: Tuple[int, int, int, int], pad: float = 0.1) -> Image.Image:
    x0, y0, x1, y1 = box
    w, h = img.size
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return img.crop((int(max(0, x0 - dx)), int(max(0, y0 - dy)),
                     int(min(w, x1 + dx)), int(min(h, y1 + dy))))


def read_done(path: Path, key: Callable[[dict], object]) -> set:
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(key(json.loads(line)))
                except json.JSONDecodeError:
                    pass  # a half-written last line from an interrupted run
    return done


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


# ----------------------------------------------------------------------------
# GPU stages
# ----------------------------------------------------------------------------

def _commit_hash(model) -> Optional[str]:
    return getattr(getattr(model, "config", None), "_commit_hash", None)


def run_captions(image_paths: Dict[int, Path], out_path: Path, log: Callable[[str], None] = print) -> None:
    """LLaVA-1.5-7B (4-bit), greedy, with per-token log-probabilities."""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

    done = read_done(out_path, lambda r: int(r["image_id"]))
    todo = [i for i in image_paths if i not in done]
    log(f"captions: {len(done)} done, {len(todo)} to go")
    if not todo:
        return
    processor = AutoProcessor.from_pretrained(LLAVA_MODEL, revision=LLAVA_REVISION)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_MODEL, revision=LLAVA_REVISION, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
    )
    model.eval()
    tok = processor.tokenizer
    prompt = f"USER: <image>\n{CAPTION_PROMPT} ASSISTANT:"
    for n, iid in enumerate(todo, 1):
        img = Image.open(image_paths[iid]).convert("RGB")
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 return_dict_in_generate=True, output_scores=True)
        gen_ids = out.sequences[0, inputs["input_ids"].shape[1]:].tolist()
        logprobs = []
        for step, tid in enumerate(gen_ids):
            lp = torch.log_softmax(out.scores[step][0].float(), dim=-1)[tid].item()
            logprobs.append(lp)
        eos = tok.eos_token_id
        if gen_ids and gen_ids[-1] == eos:
            gen_ids, logprobs = gen_ids[:-1], logprobs[:-1]
        decode = lambda ids: tok.decode(ids, skip_special_tokens=True)
        raw = decode(gen_ids)
        caption = raw.strip()
        offset = len(raw) - len(raw.lstrip())
        append_jsonl(out_path, {
            "image_id": iid, "caption": caption, "text_offset": offset,
            "token_ids": gen_ids, "token_logprobs": logprobs,
            "token_spans": token_char_offsets(decode, gen_ids),
            "model": LLAVA_MODEL, "revision": _commit_hash(model) or LLAVA_REVISION,
            "prompt": CAPTION_PROMPT, "max_new_tokens": MAX_NEW_TOKENS, "decoding": "greedy",
            "hit_max_tokens": len(gen_ids) >= MAX_NEW_TOKENS - 1,
        })
        if n % 10 == 0 or n == len(todo):
            log(f"  captions {n}/{len(todo)}")
    del model
    torch.cuda.empty_cache()


def run_evidence(
    image_paths: Dict[int, Path],
    claims_by_image: Dict[int, List[str]],
    out_path: Path,
    shifts: Iterable[str] = SHIFTS,
    log: Callable[[str], None] = print,
) -> None:
    """OWLv2 max score + box and CLIP crop/global similarity, per claim and shift."""
    import torch
    from transformers import CLIPModel, CLIPProcessor, Owlv2ForObjectDetection, Owlv2Processor

    shifts = list(shifts)
    done = read_done(out_path, lambda r: (int(r["image_id"]), r["shift"]))
    todo = [(i, s) for i in image_paths for s in shifts if claims_by_image.get(i) and (i, s) not in done]
    log(f"evidence: {len(done)} (image, shift) done, {len(todo)} to go")
    if not todo:
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    owl_proc = Owlv2Processor.from_pretrained(OWL_MODEL)
    owl = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL).to(dev).eval()
    clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clip = CLIPModel.from_pretrained(CLIP_MODEL).to(dev).eval()
    revs = {"owl": _commit_hash(owl), "clip": _commit_hash(clip)}

    for n, (iid, shift) in enumerate(todo, 1):
        img = apply_shift(Image.open(image_paths[iid]), shift)
        w, h = img.size
        cats = claims_by_image[iid]
        queries = [QUERY_TEMPLATE.format(c) for c in cats]
        with torch.inference_mode():
            enc = owl_proc(text=[queries], images=img, return_tensors="pt").to(dev)
            o = owl(**enc)
            probs = torch.sigmoid(o.logits[0].float())          # (patches, queries)
            best_s, best_p = probs.max(dim=0)
            boxes = o.pred_boxes[0].float()                      # (patches, 4) cxcywh on padded square
            crops, pix_boxes = [], []
            for q in range(len(cats)):
                b = owl_box_to_pixels(boxes[best_p[q]].tolist(), w, h)
                pix_boxes.append(b)
                crops.append(padded_crop(img, b))
            cin = clip_proc(text=queries, images=crops + [img], return_tensors="pt", padding=True).to(dev)
            co = clip(**cin)
            ie = co.image_embeds / co.image_embeds.norm(dim=-1, keepdim=True)
            te = co.text_embeds / co.text_embeds.norm(dim=-1, keepdim=True)
            sims = (ie @ te.T).float().cpu()                   # (crops+1, queries)
        for q, cat in enumerate(cats):
            append_jsonl(out_path, {
                "image_id": iid, "shift": shift, "category": cat,
                "owl_score": float(best_s[q]), "owl_box": list(pix_boxes[q]),
                "clip_crop": float(sims[q, q]), "clip_global": float(sims[len(cats), q]),
                "owl_model": OWL_MODEL, "clip_model": CLIP_MODEL, "revisions": revs,
            })
        if n % 20 == 0 or n == len(todo):
            log(f"  evidence {n}/{len(todo)}")
    del owl, clip
    if dev == "cuda":
        torch.cuda.empty_cache()
