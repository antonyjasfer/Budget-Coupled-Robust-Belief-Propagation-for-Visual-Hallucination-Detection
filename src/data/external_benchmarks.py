"""
Lightweight External Benchmark Adapters and Typed Validation Contracts.

Provides typed interfaces, validation contracts, and mapping hooks for external
benchmarks (POPE, AMBER) without fabricating compatibility.

Strictly enforces:
1. Only genuine atomic object-existence queries map to AtomicObjectExistenceClaim.
2. AMBER dimensions outside object existence (attribute, relation, etc.) are explicitly
   flagged as UNSUPPORTED_TASK_TYPE.
3. No external benchmark reproduction or downloads in Phase 9E.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union

from src.data.schemas import (
    AtomicObjectExistenceClaim,
    GroundTruthStatus,
    DatasetSource,
)


class BenchmarkTaskType(str, Enum):
    """Hallucination evaluation dimension in external benchmarks."""
    OBJECT_EXISTENCE = "object_existence"
    ATTRIBUTE_BINDING = "attribute_binding"
    SPATIAL_RELATION = "spatial_relation"
    FREE_FORM_QA = "free_form_qa"
    UNSUPPORTED_TASK_TYPE = "unsupported_task_type"


class BenchmarkAdapterStatus(str, Enum):
    """Adapter execution status for a benchmark item."""
    SUCCESS = "success"
    UNSUPPORTED = "unsupported"
    NOT_IMPLEMENTED = "not_implemented"


@dataclass
class BenchmarkMappingResult:
    """Outcome of mapping an external benchmark record to project data model."""
    benchmark_name: str
    raw_record_id: str
    task_type: BenchmarkTaskType
    adapter_status: BenchmarkAdapterStatus
    is_mappable: bool
    claim: Optional[AtomicObjectExistenceClaim] = None
    reason: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "raw_record_id": self.raw_record_id,
            "task_type": self.task_type.value,
            "adapter_status": self.adapter_status.value,
            "is_mappable": self.is_mappable,
            "claim": self.claim.to_dict() if self.claim else None,
            "reason": self.reason,
            "provenance": self.provenance,
        }


class BaseBenchmarkAdapter(ABC):
    """Abstract contract for external benchmark adapters."""
    benchmark_name: str

    @abstractmethod
    def identify_task_type(self, record: Dict[str, Any]) -> BenchmarkTaskType:
        """Inspect record and classify task type without forcing assumptions."""
        pass

    @abstractmethod
    def map_record(self, record: Dict[str, Any]) -> BenchmarkMappingResult:
        """Attempt mapping of raw record to AtomicObjectExistenceClaim."""
        pass


class POPEAdapter(BaseBenchmarkAdapter):
    """
    Lightweight adapter for POPE (Polling-based Object Probing Evaluation).
    POPE probes binary object existence in images.
    """
    benchmark_name = "POPE"

    def identify_task_type(self, record: Dict[str, Any]) -> BenchmarkTaskType:
        # POPE queries are typically "Is there a <object> in the image?"
        query = record.get("text", "") or record.get("query", "")
        if "is there" in query.lower() or "there is" in query.lower() or "present" in query.lower():
            return BenchmarkTaskType.OBJECT_EXISTENCE
        # If record explicitly specifies category or question type
        if "category" in record or "object" in record:
            return BenchmarkTaskType.OBJECT_EXISTENCE
        return BenchmarkTaskType.UNSUPPORTED_TASK_TYPE

    def map_record(self, record: Dict[str, Any]) -> BenchmarkMappingResult:
        raw_id = str(record.get("question_id") or record.get("id") or record.get("image_id", "unknown"))
        task_type = self.identify_task_type(record)

        if task_type != BenchmarkTaskType.OBJECT_EXISTENCE:
            return BenchmarkMappingResult(
                benchmark_name=self.benchmark_name,
                raw_record_id=raw_id,
                task_type=task_type,
                adapter_status=BenchmarkAdapterStatus.UNSUPPORTED,
                is_mappable=False,
                reason="POPE query is not an atomic object existence question.",
            )

        category = record.get("category") or record.get("object")
        image_id = str(record.get("image_id") or record.get("image", ""))
        
        if not category:
            return BenchmarkMappingResult(
                benchmark_name=self.benchmark_name,
                raw_record_id=raw_id,
                task_type=task_type,
                adapter_status=BenchmarkAdapterStatus.UNSUPPORTED,
                is_mappable=False,
                reason="Missing target object category in POPE record.",
            )

        # Ground truth label mapping: 'yes' -> SUPPORTED, 'no' -> HALLUCINATED
        raw_label = str(record.get("label") or record.get("ground_truth", "")).strip().lower()
        if raw_label in ("yes", "1", "true", "present"):
            gt = GroundTruthStatus.SUPPORTED
        elif raw_label in ("no", "0", "false", "absent"):
            gt = GroundTruthStatus.HALLUCINATED
        else:
            gt = GroundTruthStatus.UNKNOWN

        claim = AtomicObjectExistenceClaim(
            claim_id=f"pope_{raw_id}",
            image_id=image_id,
            object_category=category.strip().lower(),
            raw_claim_text=category.strip(),
            provenance={
                "pope_raw_id": raw_id,
                "pope_query": record.get("text", ""),
                "ground_truth": gt.value,
                "dataset_source": DatasetSource.POPE.value,
            },
        )

        return BenchmarkMappingResult(
            benchmark_name=self.benchmark_name,
            raw_record_id=raw_id,
            task_type=task_type,
            adapter_status=BenchmarkAdapterStatus.SUCCESS,
            is_mappable=True,
            claim=claim,
            provenance={"benchmark": "POPE", "record_id": raw_id},
        )


class AMBERAdapter(BaseBenchmarkAdapter):
    """
    Lightweight adapter for AMBER (An LLM-free Multidimensional Benchmark for Evaluat-ing Hallucination).
    AMBER evaluates existence, attribute, and relation hallucinations.
    """
    benchmark_name = "AMBER"

    def identify_task_type(self, record: Dict[str, Any]) -> BenchmarkTaskType:
        raw_type = str(record.get("task_type") or record.get("type") or "").strip().lower()
        if raw_type in ("existence", "object", "object_existence"):
            return BenchmarkTaskType.OBJECT_EXISTENCE
        elif raw_type in ("attribute", "attribute_binding", "color", "shape"):
            return BenchmarkTaskType.ATTRIBUTE_BINDING
        elif raw_type in ("relation", "spatial", "action"):
            return BenchmarkTaskType.SPATIAL_RELATION
        return BenchmarkTaskType.UNSUPPORTED_TASK_TYPE

    def map_record(self, record: Dict[str, Any]) -> BenchmarkMappingResult:
        raw_id = str(record.get("id") or record.get("query_id", "unknown"))
        task_type = self.identify_task_type(record)

        # Strict scope enforcement: AMBER dimensions beyond object existence are marked UNSUPPORTED_TASK_TYPE
        if task_type != BenchmarkTaskType.OBJECT_EXISTENCE:
            return BenchmarkMappingResult(
                benchmark_name=self.benchmark_name,
                raw_record_id=raw_id,
                task_type=task_type,
                adapter_status=BenchmarkAdapterStatus.UNSUPPORTED,
                is_mappable=False,
                reason=(
                    f"AMBER task type '{task_type.value}' is outside the object-existence scope "
                    "of this project. Non-existence dimensions are strictly unsupported in 9E."
                ),
            )

        category = record.get("object") or record.get("target_object")
        image_id = str(record.get("image_id") or record.get("image", ""))

        if not category:
            return BenchmarkMappingResult(
                benchmark_name=self.benchmark_name,
                raw_record_id=raw_id,
                task_type=task_type,
                adapter_status=BenchmarkAdapterStatus.UNSUPPORTED,
                is_mappable=False,
                reason="Missing target object category in AMBER existence record.",
            )

        raw_label = str(record.get("ground_truth") or record.get("label", "")).strip().lower()
        if raw_label in ("yes", "1", "present", "supported"):
            gt = GroundTruthStatus.SUPPORTED
        elif raw_label in ("no", "0", "absent", "hallucinated"):
            gt = GroundTruthStatus.HALLUCINATED
        else:
            gt = GroundTruthStatus.UNKNOWN

        claim = AtomicObjectExistenceClaim(
            claim_id=f"amber_{raw_id}",
            image_id=image_id,
            object_category=category.strip().lower(),
            raw_claim_text=category.strip(),
            provenance={
                "amber_raw_id": raw_id,
                "task_type": task_type.value,
                "ground_truth": gt.value,
                "dataset_source": DatasetSource.COCO.value,
            },
        )

        return BenchmarkMappingResult(
            benchmark_name=self.benchmark_name,
            raw_record_id=raw_id,
            task_type=task_type,
            adapter_status=BenchmarkAdapterStatus.SUCCESS,
            is_mappable=True,
            claim=claim,
            provenance={"benchmark": "AMBER", "record_id": raw_id},
        )

