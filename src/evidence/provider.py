"""
Evidence provider interface / protocol.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.data.schemas import AtomicObjectExistenceClaim
from src.evidence.schemas import RawEvidenceRecord


class EvidenceProvider(ABC):
    """
    Abstract interface for obtaining raw visual evidence for atomic claims.
    """

    @abstractmethod
    def get_evidence(
        self,
        image_id: str,
        claim: AtomicObjectExistenceClaim,
        view_id: str = "original",
    ) -> RawEvidenceRecord:
        """
        Retrieve raw evidence for a single claim on a specified image view.
        """
        pass

    def get_batch_evidence(
        self,
        image_id: str,
        claims: List[AtomicObjectExistenceClaim],
        view_id: str = "original",
    ) -> List[RawEvidenceRecord]:
        """
        Retrieve raw evidence for a list of claims on a specified image view.
        """
        return [self.get_evidence(image_id, claim, view_id=view_id) for claim in claims]
