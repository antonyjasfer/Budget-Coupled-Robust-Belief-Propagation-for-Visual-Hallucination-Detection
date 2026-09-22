"""Unit tests for non-PGM logistic evidence baseline models."""

import numpy as np
import pytest

from src.calibration.evidence_models import (
    LogisticEvidenceModel,
    extract_evidence_features,
)


def test_feature_extraction_dimensions():
    """Verify feature extractor output shapes and types."""
    records = [
        {"detector_score": 0.8, "clip_score": 0.25},
        {"detector_score": 0.1, "clip_score": 0.15},
    ]

    # Detector only -> 1 feature
    x_det, names_det = extract_evidence_features(records, feature_type="detector_only")
    assert x_det.shape == (2, 1)
    assert names_det == ["detector_score"]

    # CLIP only -> 1 feature
    x_clip, names_clip = extract_evidence_features(records, feature_type="clip_only")
    assert x_clip.shape == (2, 1)
    assert names_clip == ["clip_score"]

    # Combined -> 2 features
    x_comb, names_comb = extract_evidence_features(records, feature_type="combined")
    assert x_comb.shape == (2, 2)
    assert names_comb == ["detector_score", "clip_score"]

    # Combined interaction -> 3 features
    x_inter, names_inter = extract_evidence_features(records, feature_type="combined_interaction")
    assert x_inter.shape == (2, 3)
    assert "detector_x_clip" in names_inter


def test_logistic_model_fit_and_predict():
    """Verify standard logistic regression fitting on synthetic separable data."""
    # Synthetic data: high detector/clip -> hallucinated = 0 (supported), low -> hallucinated = 1
    # Recall label = 1 means HALLUCINATED
    rng = np.random.RandomState(42)
    n = 60
    # Class 0: high scores (detector ~ 0.8, clip ~ 0.3)
    c0_det = rng.normal(0.8, 0.05, n // 2)
    c0_clip = rng.normal(0.3, 0.05, n // 2)
    # Class 1: low scores (detector ~ 0.2, clip ~ 0.1)
    c1_det = rng.normal(0.2, 0.05, n // 2)
    c1_clip = rng.normal(0.1, 0.05, n // 2)

    records = []
    for d, c in zip(c0_det, c0_clip):
        records.append({"detector_score": d, "clip_score": c, "label": 0})
    for d, c in zip(c1_det, c1_clip):
        records.append({"detector_score": d, "clip_score": c, "label": 1})

    train_records = records[:40]
    val_records = records[40:]

    model = LogisticEvidenceModel(feature_type="combined")
    model.fit(train_records, val_records=val_records)

    assert model.fitted
    assert model.coefficients is not None
    assert len(model.coefficients) == 2
    assert model.intercept is not None

    # Predict proba
    probs = model.predict_proba(val_records)
    assert len(probs) == len(val_records)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Low detector score should have higher hallucination probability
    test_low = [{"detector_score": 0.1, "clip_score": 0.05}]
    test_high = [{"detector_score": 0.9, "clip_score": 0.35}]
    p_low = model.predict_proba(test_low)[0]
    p_high = model.predict_proba(test_high)[0]
    assert p_low > p_high


def test_validation_hyperparameter_selection():
    """Verify that regularizer C is selected using validation performance."""
    rng = np.random.RandomState(123)
    train_records = [
        {"detector_score": float(d), "clip_score": float(c), "label": int(l)}
        for d, c, l in zip(rng.rand(30), rng.rand(30), rng.choice([0, 1], 30))
    ]
    val_records = [
        {"detector_score": float(d), "clip_score": float(c), "label": int(l)}
        for d, c, l in zip(rng.rand(15), rng.rand(15), rng.choice([0, 1], 15))
    ]

    model = LogisticEvidenceModel(feature_type="detector_only")
    model.fit(train_records, val_records=val_records)
    assert model.best_c in [0.01, 0.1, 1.0, 10.0, 100.0]


def test_tiny_or_single_class_fallback():
    """Verify graceful fallback without crashing when only one class is present."""
    single_class_train = [
        {"detector_score": 0.8, "clip_score": 0.3, "label": 0},
        {"detector_score": 0.7, "clip_score": 0.25, "label": 0},
    ]
    model = LogisticEvidenceModel(feature_type="combined")
    model.fit(single_class_train)

    assert model.fitted
    # Must predict fallback without NaN
    probs = model.predict_proba(single_class_train)
    assert np.all(np.isfinite(probs))
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_serialization_and_reconstruction():
    """Verify model to_dict and from_dict integrity."""
    model = LogisticEvidenceModel(feature_type="combined")
    train_records = [
        {"detector_score": 0.8, "clip_score": 0.3, "label": 0},
        {"detector_score": 0.2, "clip_score": 0.1, "label": 1},
        {"detector_score": 0.7, "clip_score": 0.28, "label": 0},
        {"detector_score": 0.3, "clip_score": 0.12, "label": 1},
    ]
    model.fit(train_records)
    d = model.to_dict()

    m_recon = LogisticEvidenceModel.from_dict(d)
    assert m_recon.fitted
    assert m_recon.feature_type == "combined"
    assert m_recon.model_hash == model.model_hash

    p1 = model.predict_proba(train_records)
    p2 = m_recon.predict_proba(train_records)
    assert np.allclose(p1, p2)
