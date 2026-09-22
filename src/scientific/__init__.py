"""
Milestone 9: Final Scientific Validation, Results Consolidation, and Research-Ready Output.
"""

from src.scientific.gate import FinalDatasetGate, GateCheckResult, GateStatus
from src.scientific.consolidator import M9ResultsConsolidator, M9ConsolidatedData
from src.scientific.statistics import (
    compute_m9_paired_bootstrap,
    compute_correlation_hypotheses,
    categorize_errors,
    select_m9_case_studies,
)
from src.scientific.tables import generate_all_m9_tables
from src.scientific.figures import generate_all_m9_figures
from src.scientific.reporting import generate_m9_research_package

__all__ = [
    "FinalDatasetGate",
    "GateCheckResult",
    "GateStatus",
    "M9ResultsConsolidator",
    "M9ConsolidatedData",
    "compute_m9_paired_bootstrap",
    "compute_correlation_hypotheses",
    "categorize_errors",
    "select_m9_case_studies",
    "generate_all_m9_tables",
    "generate_all_m9_figures",
    "generate_m9_research_package",
]
