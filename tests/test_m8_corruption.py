"""Tests for Milestone 8 Visual Corruption Transformations."""

import pytest
import numpy as np
from PIL import Image

from src.experiments.corruption import (
    apply_visual_corruption,
    apply_image_corruption,
    create_corrupted_claim_record,
)


def test_gaussian_blur_corruption():
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    corrupted = apply_visual_corruption(img, "gaussian_blur", "heavy", seed=42)
    assert corrupted.size == (64, 64)
    assert corrupted.mode == "RGB"


def test_jpeg_compression_corruption():
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    corrupted = apply_visual_corruption(img, "jpeg_compression", "medium", seed=42)
    assert corrupted.size == (64, 64)


def test_additive_noise_deterministic_seed():
    img = Image.new("RGB", (32, 32), color=(100, 100, 100))
    c1 = apply_visual_corruption(img, "additive_noise", "medium", seed=123)
    c2 = apply_visual_corruption(img, "additive_noise", "medium", seed=123)
    c3 = apply_visual_corruption(img, "additive_noise", "medium", seed=999)

    arr1 = np.array(c1)
    arr2 = np.array(c2)
    arr3 = np.array(c3)

    assert np.array_equal(arr1, arr2)
    assert not np.array_equal(arr1, arr3)


def test_center_occlusion_corruption():
    img = Image.new("RGB", (50, 50), color=(255, 255, 255))
    corrupted = apply_visual_corruption(img, "center_occlusion", "heavy", seed=42)
    arr = np.array(corrupted)
    # Center pixel should be neutral grey patch (128, 128, 128)
    assert np.all(arr[25, 25] == 128)
