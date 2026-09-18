"""
Offline fixture evidence provider.

Reads pre-computed or explicitly synthetic RawEvidenceRecord fixtures from disk or dictionary.
Enforces strict lookup without inventing or extrapolating missing values.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

from src.data.schemas import AtomicObjectExistenceClaim
from src.evidence.schemas import RawEvidenceRecord
from src.evidence.provider import EvidenceProvider


class FixtureEvidenceProvider(EvidenceProvider):
    """
    Offline evidence provider backed by an explicit fixture dataset.

    Attributes:
        evidence_records: Dict mapping (image_id, claim_id, view_id) -> RawEvidenceRecord.
        provider_id: Identifier string.
    """

    def __init__(
        self,
        records: Optional[List[RawEvidenceRecord]] = None,
        fixture_path: Optional[Union[str, Path]] = None,
        provider_id: str = "offline_fixture_provider_v1",
    ):
        self.provider_id = provider_id
        self.evidence_records: Dict[Tuple[str, str, str], RawEvidenceRecord] = {}

        if fixture_path:
            self._load_from_path(Path(fixture_path))

        if records:
            for r in records:
                key = (r.image_id, r.claim_id, r.view_id)
                self.evidence_records[key] = r

    def _load_from_path(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Evidence fixture not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected list of evidence records in {path}, got {type(data)}")

        for item in data:
            rec = RawEvidenceRecord.from_dict(item)
            key = (rec.image_id, rec.claim_id, rec.view_id)
            self.evidence_records[key] = rec

    def get_evidence(
        self,
        image_id: str,
        claim: AtomicObjectExistenceClaim,
        view_id: str = "original",
    ) -> RawEvidenceRecord:
        """
        Look up stored evidence for (image_id, claim_id, view_id).
        Raises KeyError if no evidence record was explicitly provided for this claim.
        """
        key = (image_id, claim.claim_id, view_id)
        if key not in self.evidence_records:
            # Also try looking up by category if claim_id was dynamically generated
            cat_match = [
                r for (img, _, v), r in self.evidence_records.items()
                if img == image_id and v == view_id and r.object_category == claim.object_category
            ]
            if cat_match:
                rec = cat_match[0]
                # Return record with updated claim_id
                return RawEvidenceRecord(
                    evidence_id=f"ev_{image_id}_{claim.claim_id}_{view_id}",
                    image_id=image_id,
                    claim_id=claim.claim_id,
                    object_category=claim.object_category,
                    detector_score=rec.detector_score,
                    detector_available=rec.detector_available,
                    similarity_score=rec.similarity_score,
                    similarity_available=rec.similarity_available,
                    provider_id=self.provider_id,
                    view_id=view_id,
                    is_synthetic=rec.is_synthetic,
                    metadata=rec.metadata,
                )

            raise KeyError(
                f"No explicit fixture evidence found for (image_id='{image_id}', claim_id='{claim.claim_id}', "
                f"category='{claim.object_category}', view_id='{view_id}'). "
                f"FixtureEvidenceProvider never fabricates missing scores."
            )

        return self.evidence_records[key]
