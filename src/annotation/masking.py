"""
Evidence Masking and Annotation Template Generator for M7.

STRICT METHODOLOGICAL GUARANTEES:
1. Strips all model evidence (detector_score, clip_score, detector_available, similarity_available).
2. Strips model confidence, posteriors, and PGM/robust-BP potentials (theta, epsilon, J).
3. Strips predictions, decisions, abstain flags, and split partition assignments.
4. Strips full VLM caption by default to prevent linguistic confirmation bias.
5. Only exposes visual target (category, surface span, minimal sentence context) and image reference.
"""

from typing import List, Dict, Optional, Any, Union
import re

from src.annotation.schemas import HumanAnnotationTask, M7ClaimRecord
from src.evidence.schemas import ClaimLevelEvidenceRecord

FORBIDDEN_EVIDENCE_FIELDS = {
    "detector_score",
    "detector_available",
    "detector_model",
    "detector_revision",
    "detector_configuration",
    "clip_score",
    "similarity_available",
    "clip_model",
    "clip_revision",
    "clip_prompt_template",
    "preprocessing_configuration",
    "posterior",
    "posterior_bounds",
    "theta",
    "epsilon",
    "J",
    "prediction",
    "decision",
    "abstain",
    "split",
    "vlm_model",
    "vlm_revision",
}


def extract_minimal_context(caption: str, surface_form: str, max_words: int = 15) -> Optional[str]:
    """
    Extract a minimal sentence or local phrase context around the surface form.

    Avoids exposing full complex multi-sentence paragraphs that bias the human annotator.

    Args:
        caption: Original full caption.
        surface_form: Mentioned surface span.
        max_words: Maximum words in extracted context window.

    Returns:
        Minimal context snippet or None.
    """
    if not caption or not surface_form:
        return None

    # Find the sentence containing the surface form
    sentences = re.split(r'(?<=[.!?])\s+', caption)
    for s in sentences:
        if surface_form.lower() in s.lower():
            words = s.strip().split()
            if len(words) <= max_words:
                return s.strip()
            # If sentence is long, slice window around occurrence
            lower_words = [w.lower() for w in words]
            target_words = surface_form.lower().split()
            for idx in range(len(words) - len(target_words) + 1):
                if lower_words[idx:idx + len(target_words)] == target_words:
                    start = max(0, idx - 4)
                    end = min(len(words), idx + len(target_words) + 4)
                    prefix = "..." if start > 0 else ""
                    suffix = "..." if end < len(words) else ""
                    return f"{prefix}{' '.join(words[start:end])}{suffix}"
            return s.strip()

    # Fallback to bounded slice
    words = caption.split()
    if len(words) <= max_words:
        return caption.strip()
    return " ".join(words[:max_words]) + "..."


def create_masked_annotation_task(
    record: Union[M7ClaimRecord, ClaimLevelEvidenceRecord],
    image_path_lookup: Optional[Dict[str, str]] = None,
    include_minimal_context: bool = True,
    annotator_tag: Optional[str] = None,
) -> HumanAnnotationTask:
    """
    Convert an M7ClaimRecord or ClaimLevelEvidenceRecord into a strictly masked HumanAnnotationTask.

    Args:
        record: Master claim or evidence record.
        image_path_lookup: Optional dict mapping image_id to local image path.
        include_minimal_context: If True, provides minimal sentence snippet; never full caption.
        annotator_tag: Optional tag (e.g., 'annotator_A').

    Returns:
        HumanAnnotationTask free of any model evidence or bias.
    """
    image_id = record.image_id
    claim_id = record.claim_id
    category = record.object_category
    surface_form = record.text_span or category

    image_path = (
        image_path_lookup.get(image_id, f"images/{image_id}.jpg")
        if image_path_lookup
        else f"images/{image_id}.jpg"
    )

    minimal_context = None
    if include_minimal_context and record.caption:
        minimal_context = extract_minimal_context(record.caption, surface_form)

    task_prefix = f"{annotator_tag}_" if annotator_tag else ""
    task_id = f"task_{task_prefix}{claim_id}"

    task = HumanAnnotationTask(
        task_id=task_id,
        claim_id=claim_id,
        image_id=image_id,
        image_path=image_path,
        object_category=category,
        surface_form=surface_form,
        minimal_context=minimal_context,
        metadata={"annotator_tag": annotator_tag} if annotator_tag else {},
    )

    # Sanity check: Ensure no forbidden field leaked into serialized representation
    audit_masked_task(task)
    return task


def audit_masked_task(task: HumanAnnotationTask) -> None:
    """
    Machine-checkable audit that asserts forbidden fields are absent from the task dictionary.

    Raises:
        ValueError: If any forbidden evidence or model field is detected.
    """
    d = task.to_dict()
    for forbidden in FORBIDDEN_EVIDENCE_FIELDS:
        if forbidden in d:
            raise ValueError(f"Forbidden field '{forbidden}' found in HumanAnnotationTask!")
        if forbidden in d.get("metadata", {}):
            raise ValueError(f"Forbidden field '{forbidden}' found in HumanAnnotationTask metadata!")
