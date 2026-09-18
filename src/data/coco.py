"""
Local COCO metadata adapter.

Parses local COCO instance annotations and image metadata.
Extracts canonical ImageRecords, CategoryRegistry, and per-image annotated category presence.

CRITICAL METHODOLOGICAL SAFEGUARD:
Missing category annotations indicate ABSENCE OF ANNOTATION in the COCO dataset,
NOT definitive physical object absence in the image. This module strictly treats
unannotated categories as unobserved evidence, never as negative ground truth labels.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union
import json

from src.data.schemas import ImageRecord, DatasetSource


@dataclass
class COCOCategory:
    """COCO Category entry."""
    category_id: int
    name: str
    supercategory: str

    @property
    def canonical_name(self) -> str:
        return self.name.strip().lower()


@dataclass
class COCOImageEvidence:
    """
    Evidence of annotated object categories for a single COCO image.

    Attributes:
        image_id: Canonical image ID ('coco_{coco_id}').
        coco_id: Original COCO integer ID.
        present_category_ids: Set of category IDs explicitly annotated in the image.
        present_category_names: Set of canonical category names explicitly annotated.
        instance_counts: Map of category_id -> count of annotated instances in the image.
        bounding_boxes: Map of category_id -> list of [x, y, w, h] boxes.
    """
    image_id: str
    coco_id: int
    present_category_ids: Set[int] = field(default_factory=set)
    present_category_names: Set[str] = field(default_factory=set)
    instance_counts: Dict[int, int] = field(default_factory=dict)
    bounding_boxes: Dict[int, List[List[float]]] = field(default_factory=dict)

    def is_annotated_present(self, category_name_or_id: Union[str, int]) -> bool:
        """Check if category has at least one explicit bounding box annotation."""
        if isinstance(category_name_or_id, int):
            return category_name_or_id in self.present_category_ids
        return category_name_or_id.strip().lower() in self.present_category_names


class COCOMetadataAdapter:
    """
    Adapter for reading and indexing local COCO instance annotations.

    Attributes:
        annotation_path: Path to local COCO instances JSON file.
        images: Dict mapping canonical image_id -> ImageRecord.
        categories: Dict mapping category_id -> COCOCategory.
        name_to_category: Dict mapping canonical category name -> COCOCategory.
        evidence: Dict mapping canonical image_id -> COCOImageEvidence.
        provenance: Metadata describing the loaded annotation file.
    """

    def __init__(self, annotation_path: Union[str, Path]):
        self.annotation_path = Path(annotation_path)
        self.images: Dict[str, ImageRecord] = {}
        self.categories: Dict[int, COCOCategory] = {}
        self.name_to_category: Dict[str, COCOCategory] = {}
        self.evidence: Dict[str, COCOImageEvidence] = {}
        self.provenance: Dict[str, Any] = {}

        self._load_and_parse()

    def _load_and_parse(self):
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {self.annotation_path}")

        try:
            with open(self.annotation_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse COCO JSON at {self.annotation_path}: {e}")

        # Record provenance
        self.provenance = {
            "source_file": str(self.annotation_path.name),
            "info": data.get("info", {}),
            "num_images_loaded": len(data.get("images", [])),
            "num_annotations_loaded": len(data.get("annotations", [])),
            "num_categories_loaded": len(data.get("categories", [])),
        }

        # 1. Parse categories
        for cat in data.get("categories", []):
            cid = int(cat["id"])
            cname = str(cat["name"]).strip().lower()
            supercat = str(cat.get("supercategory", ""))
            coco_cat = COCOCategory(category_id=cid, name=cname, supercategory=supercat)
            self.categories[cid] = coco_cat
            self.name_to_category[cname] = coco_cat

        # 2. Parse images
        for img in data.get("images", []):
            coco_id = int(img["id"])
            canonical_id = f"coco_{coco_id}"
            file_name = img.get("file_name", f"{coco_id:012d}.jpg")
            img_rec = ImageRecord(
                image_id=canonical_id,
                dataset_source=DatasetSource.COCO,
                file_name=file_name,
                width=img.get("width"),
                height=img.get("height"),
                coco_id=coco_id,
                metadata={
                    "coco_url": img.get("coco_url"),
                    "date_captured": img.get("date_captured"),
                    "license": img.get("license"),
                },
            )
            self.images[canonical_id] = img_rec
            self.evidence[canonical_id] = COCOImageEvidence(
                image_id=canonical_id,
                coco_id=coco_id,
            )

        # 3. Parse annotations
        for ann in data.get("annotations", []):
            coco_id = int(ann["image_id"])
            canonical_id = f"coco_{coco_id}"
            cat_id = int(ann["category_id"])

            if canonical_id not in self.evidence:
                continue

            ev = self.evidence[canonical_id]
            ev.present_category_ids.add(cat_id)
            if cat_id in self.categories:
                ev.present_category_names.add(self.categories[cat_id].canonical_name)

            ev.instance_counts[cat_id] = ev.instance_counts.get(cat_id, 0) + 1
            if "bbox" in ann:
                if cat_id not in ev.bounding_boxes:
                    ev.bounding_boxes[cat_id] = []
                ev.bounding_boxes[cat_id].append(ann["bbox"])

    def get_image_record(self, image_id_or_coco_id: Union[str, int]) -> Optional[ImageRecord]:
        """Get canonical ImageRecord by string image_id ('coco_123') or integer coco_id."""
        if isinstance(image_id_or_coco_id, int):
            canonical_id = f"coco_{image_id_or_coco_id}"
        else:
            canonical_id = image_id_or_coco_id
        return self.images.get(canonical_id)

    def get_evidence(self, image_id_or_coco_id: Union[str, int]) -> Optional[COCOImageEvidence]:
        """Get COCOImageEvidence for an image."""
        if isinstance(image_id_or_coco_id, int):
            canonical_id = f"coco_{image_id_or_coco_id}"
        else:
            canonical_id = image_id_or_coco_id
        return self.evidence.get(canonical_id)

    def to_manifest_entries(self) -> List[Any]:
        """Convert loaded COCO images into DatasetManifestEntry records."""
        from src.data.schemas import DatasetManifestEntry
        return [DatasetManifestEntry(image=img_rec) for img_rec in self.images.values()]


def load_coco_instances(annotation_path: Union[str, Path]) -> COCOMetadataAdapter:
    """Convenience helper to load COCO instances from JSON path."""
    return COCOMetadataAdapter(annotation_path)

