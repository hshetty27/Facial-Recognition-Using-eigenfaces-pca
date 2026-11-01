"""
Phase 1 environment checks: required libraries and project modules import.

Mathematical tests will be added once the PCA implementation exists.
"""

import importlib

import pytest


def test_numpy_imports():
    import numpy as np

    assert hasattr(np.linalg, "svd")


def test_matplotlib_imports():
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend, works without a display
    import matplotlib.pyplot  # noqa: F401


def test_pillow_imports():
    from PIL import Image  # noqa: F401


@pytest.mark.parametrize(
    "module",
    [
        "src",
        "src.data_loader",
        "src.preprocessing",
        "src.pca",
        "src.projection",
        "src.recognition",
        "src.visualization",
    ],
)
def test_project_modules_import(module):
    importlib.import_module(module)
