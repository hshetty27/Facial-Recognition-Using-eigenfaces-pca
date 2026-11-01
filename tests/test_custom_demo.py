"""
Phase 5 tests: custom-dataset demo, deterministic file ordering, EXIF orientation,
split helper, numerical rank, and the demo command line.

Synthetic "people" are generated as images on disk (no real photos needed).
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.custom_demo import (
    check_no_overlap, class_mean_projections, evaluate_split, face_space_residual, leave_one_out,
    load_labelled_folder, load_model, nearest_class_mean, prepare_custom_split, recognize_image,
    save_model, train_model, truncate_model,
)
from src.data_loader import discover_images, load_image, split_indices, train_test_split
from src.pca import center_data, compute_mean_face, compute_svd, numerical_rank
from src.preprocessing import IMAGE_SHAPE, preprocess_image
from src.projection import project_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_people(root: Path, names: list[str], n_images: int, seed: int = 0, size=(92, 112)) -> Path:
    """Each person = a distinct smooth random pattern; each image = pattern + small noise, saved as JPG/PNG."""
    rng = np.random.default_rng(seed)
    for person_index, name in enumerate(names):
        base = rng.random((14, 12))
        pattern = np.kron(base, np.ones((8, 8)))[:112, :92]  # blocky "face", 112 x 92
        folder = root / name
        folder.mkdir(parents=True)
        for i in range(1, n_images + 1):
            image = np.clip(pattern + 0.03 * rng.standard_normal(pattern.shape), 0, 1)
            rgb = np.stack([image * 255] * 3, axis=-1).astype(np.uint8)
            suffix = ".jpg" if i % 2 else ".png"
            Image.fromarray(rgb).resize(size).save(folder / f"{i:02d}{suffix}")
    return root


# ---------- deterministic ordering and loading ----------

def test_natural_order_of_people_and_files(tmp_path):
    for name in ["bob", "Alice", "s10", "s2", "Charlie"]:
        (tmp_path / name).mkdir()
        for f in ["10.png", "2.png", "1.png"]:
            Image.new("L", (4, 4)).save(tmp_path / name / f)
    (tmp_path / ".hidden").mkdir()
    Image.new("L", (4, 4)).save(tmp_path / ".hidden" / "1.png")
    paths, labels = discover_images(tmp_path)
    assert list(dict.fromkeys(labels)) == ["Alice", "bob", "Charlie", "s2", "s10"]
    assert [p.name for p in paths[:3]] == ["1.png", "2.png", "10.png"]


def test_exif_orientation_is_applied(tmp_path):
    image = Image.new("RGB", (40, 20), color=(200, 10, 10))
    exif = image.getexif()
    exif[0x0112] = 6  # "rotate 90 degrees": stored landscape, displayed portrait
    image.save(tmp_path / "phone.jpg", exif=exif)
    assert Image.open(tmp_path / "phone.jpg").size == (40, 20)
    assert load_image(tmp_path / "phone.jpg").size == (20, 40)


def test_custom_images_use_project_preprocessing(tmp_path):
    make_people(tmp_path, ["Alice", "Bob"], 3)
    X, labels, paths, _ = load_labelled_folder(tmp_path)
    assert X.shape == (IMAGE_SHAPE[0] * IMAGE_SHAPE[1], 6)
    assert labels.tolist() == ["Alice"] * 3 + ["Bob"] * 3
    for j, path in enumerate(paths):
        assert np.array_equal(X[:, j], preprocess_image(load_image(path), IMAGE_SHAPE))


def test_exif_orientation_then_crop_keeps_face_upright(tmp_path):
    """Phone photo stored sideways (672 x 448) with EXIF tag 6: dark top half must end up at the top."""
    upright = np.full((672, 448), 255, dtype=np.uint8)
    upright[:300, :] = 0                                       # dark top, as displayed
    stored = Image.fromarray(upright).transpose(Image.Transpose.ROTATE_90).convert("RGB")
    exif = stored.getexif()
    exif[0x0112] = 6
    stored.save(tmp_path / "phone.jpg", exif=exif, quality=95)
    image = load_image(tmp_path / "phone.jpg")
    assert image.size == (448, 672)
    I = preprocess_image(image).reshape(IMAGE_SHAPE)
    assert I[:20].mean() < 0.1 and I[-20:].mean() > 0.9


def test_portrait_custom_photos_use_the_same_function(tmp_path):
    """448 x 672 photos: training columns, held-out test vectors and recognize_image all use preprocess_image."""
    make_people(tmp_path, ["Alice", "Bob", "Cara"], 4, size=(448, 672))
    split = prepare_custom_split(tmp_path, n_test_per_person=1, seed=0)
    for j, path in enumerate(split["train_paths"]):
        assert np.array_equal(split["X_train"][:, j], preprocess_image(load_image(path), IMAGE_SHAPE))
    for j, path in enumerate(split["test_paths"]):
        assert np.array_equal(split["X_test"][:, j], preprocess_image(load_image(path), IMAGE_SHAPE))
    model = train_model(split["X_train"], split["labels_train"], split["train_paths"], k=5)
    for path in split["test_paths"]:
        assert np.array_equal(recognize_image(model, path)["x"], preprocess_image(load_image(path), IMAGE_SHAPE))


def test_aspect_ratio_warning(tmp_path):
    make_people(tmp_path, ["Wide"], 2, size=(200, 100))  # landscape images
    _, _, _, warnings = load_labelled_folder(tmp_path)
    assert len(warnings) == 2


# ---------- split ----------

def test_split_indices_matches_train_test_split():
    labels = np.array([f"p{i // 10}" for i in range(40)])
    X = np.random.default_rng(0).random((5, 40))
    is_test = split_indices(labels, 3, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, labels, 3, seed=42)
    assert np.array_equal(X_train, X[:, ~is_test]) and np.array_equal(X_test, X[:, is_test])
    assert is_test.sum() == 12


def test_split_zero_test_images_and_too_few_images():
    labels = np.array(["a", "a", "b", "b"])
    assert not split_indices(labels, 0, seed=1).any()
    with pytest.raises(ValueError):
        split_indices(labels, 2, seed=1)  # nobody would keep a training image


def test_custom_split_is_disjoint_and_reproducible(tmp_path):
    make_people(tmp_path, ["Alice", "Bob", "Cara"], 6)
    a = prepare_custom_split(tmp_path, n_test_per_person=2, seed=7)
    b = prepare_custom_split(tmp_path, n_test_per_person=2, seed=7)
    assert a["test_paths"] == b["test_paths"]
    assert not set(a["train_paths"]) & set(a["test_paths"])
    assert len(a["train_paths"]) == 12 and len(a["test_paths"]) == 6
    for person in ["Alice", "Bob", "Cara"]:
        assert np.sum(a["labels_test"] == person) == 2


def test_identical_file_in_train_and_test_is_rejected(tmp_path):
    train_dir = make_people(tmp_path / "train", ["Alice", "Bob"], 4)
    test_dir = tmp_path / "test" / "Alice"
    test_dir.mkdir(parents=True)
    shutil.copy(train_dir / "Alice" / "02.png", test_dir / "renamed_copy.png")
    with pytest.raises(ValueError, match="identical"):
        prepare_custom_split(train_dir, test_dir=tmp_path / "test")
    with pytest.raises(ValueError):
        check_no_overlap([train_dir / "Alice" / "02.png"], [test_dir / "renamed_copy.png"])


# ---------- training uses the existing PCA ----------

def test_numerical_rank_of_centered_data():
    X = np.random.default_rng(0).random((50, 8))
    X_c = center_data(X, compute_mean_face(X))
    _, S, _ = compute_svd(X_c)
    assert numerical_rank(S, X_c.shape) == 7          # N - 1
    assert numerical_rank(np.zeros(3), (5, 3)) == 0


def test_train_model_is_the_project_pca(tmp_path):
    make_people(tmp_path, ["Alice", "Bob", "Cara", "Dan"], 5)
    split = prepare_custom_split(tmp_path, n_test_per_person=1, seed=0)
    model = train_model(split["X_train"], split["labels_train"], split["train_paths"], k=6)
    X_train = split["X_train"]
    mu = compute_mean_face(X_train)
    X_c = center_data(X_train, mu)
    U, _, _ = compute_svd(X_c)
    assert np.allclose(model["mean_face"], mu)
    assert np.allclose(model["U_k"], U[:, :6])
    assert np.allclose(model["Z_train"], project_data(X_c, U[:, :6]))
    assert np.allclose(model["U_k"].T @ model["U_k"], np.eye(6), atol=1e-10)


def test_k_is_reduced_to_rank(tmp_path):
    make_people(tmp_path, ["Alice", "Bob"], 4)       # 8 training images -> rank 7
    split = prepare_custom_split(tmp_path, n_test_per_person=0, seed=0)
    model = train_model(split["X_train"], split["labels_train"], split["train_paths"], k=30)
    assert (model["k_requested"], model["rank"], model["k"]) == (30, 7, 7)
    assert model["U_k"].shape == (IMAGE_SHAPE[0] * IMAGE_SHAPE[1], 7)


# ---------- recognition ----------

@pytest.mark.parametrize("names", [["Alice", "Bob", "Cara"], ["P1", "P2", "P3", "P4", "P5"]])
def test_recognition_of_held_out_images(tmp_path, names):
    """Any number of identities; held-out synthetic images are recognised."""
    make_people(tmp_path, names, 6)
    split = prepare_custom_split(tmp_path, n_test_per_person=2, seed=3)
    model = train_model(split["X_train"], split["labels_train"], split["train_paths"], k=10)
    evaluation = evaluate_split(model, split)
    assert evaluation["accuracy"] == 1.0
    for result, truth in zip(evaluation["results"], split["labels_test"]):
        assert result["predicted"] == truth
        assert Path(result["nearest_file"]).parent.name == truth
        assert np.isclose(min(result["per_identity"].values()), result["distance"])
        assert set(result["per_identity"]) == set(names)
        assert not result["in_training_set"]


def test_training_image_is_flagged(tmp_path):
    make_people(tmp_path, ["Alice", "Bob"], 4)
    split = prepare_custom_split(tmp_path, n_test_per_person=1, seed=0)
    model = train_model(split["X_train"], split["labels_train"], split["train_paths"], k=3)
    result = recognize_image(model, split["train_paths"][0])
    assert result["in_training_set"]
    assert result["distance"] < 1e-9


def test_save_and_load_model_roundtrip(tmp_path):
    make_people(tmp_path / "data", ["Alice", "Bob", "Cara"], 5)
    split = prepare_custom_split(tmp_path / "data", n_test_per_person=1, seed=0)
    model = train_model(split["X_train"], split["labels_train"], split["train_paths"], k=5)
    save_model(model, tmp_path / "model.npz")
    loaded = load_model(tmp_path / "model.npz")
    assert loaded["k"] == 5 and loaded["image_shape"] == IMAGE_SHAPE
    for path in split["test_paths"]:
        a, b = recognize_image(model, path), recognize_image(loaded, path)
        assert a["predicted"] == b["predicted"] and np.isclose(a["distance"], b["distance"])


# ---------- command line ----------

def test_demo_cli_train_and_recognize(tmp_path):
    make_people(tmp_path / "data", ["Alice", "Bob", "Cara", "Dan"], 5)
    model = tmp_path / "model.npz"
    common = [sys.executable, "demo.py", "--no-show", "--model", str(model), "--data-dir", str(tmp_path / "data")]
    train = subprocess.run(common + ["--k", "8", "--n-test", "1", "train"],
                           cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert train.returncode == 0, train.stderr
    assert "accuracy on held-out images" in train.stdout
    image = tmp_path / "data" / "Cara" / "03.jpg"
    recognize = subprocess.run(common + ["recognize", str(image)], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert recognize.returncode == 0, recognize.stderr
    assert "Predicted identity:" in recognize.stdout
    assert "not a probability" in recognize.stdout
    assert "%" not in recognize.stdout.split("Predicted identity:")[1]  # no fake confidence percentage


def test_demo_cli_train_all_data(tmp_path):
    """--all-data (after or before 'train') uses every image for PCA: rank = N - 1, no held-out images."""
    make_people(tmp_path / "data", ["Alice", "Bob", "Cara", "Dan"], 5)
    model = tmp_path / "model.npz"
    common = [sys.executable, "demo.py", "--no-show", "--model", str(model), "--data-dir", str(tmp_path / "data")]
    for args in (["--k", "30", "train", "--all-data"], ["--all-data", "--k", "30", "train"]):
        train = subprocess.run(common + args, cwd=PROJECT_ROOT, capture_output=True, text=True)
        assert train.returncode == 0, train.stderr
        assert "No held-out test images: all 20 images used for PCA" in train.stdout
        loaded = load_model(model)
        assert loaded["Z_train"].shape[1] == 20 and loaded["rank"] == 19 and loaded["k"] == 19
        assert loaded["labels_test"].size == 0


def test_demo_cli_model_records_training_files_and_preprocessing(tmp_path):
    """The saved model identifies exactly which files built it (SHA-256) and how they were preprocessed."""
    from src.custom_demo import file_fingerprint
    make_people(tmp_path / "data", ["Alice", "Bob", "Cara"], 4)
    model = tmp_path / "model.npz"
    train = subprocess.run([sys.executable, "demo.py", "--no-show", "--model", str(model), "--data-dir",
                            str(tmp_path / "data"), "--k", "5", "train", "--all-data"],
                           cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert train.returncode == 0, train.stderr
    loaded = load_model(model)
    assert loaded["train_sha256"].tolist() == [file_fingerprint(Path(f)) for f in loaded["train_files"]]
    assert "78x64" in str(loaded["preprocessing"]) and "grayscale" in str(loaded["preprocessing"])


def test_audit_dataset_verification(tmp_path):
    """experiments/custom_audit: counts per identity, byte-identical duplicates, mystery files that are training files."""
    from experiments.custom_audit import verify_dataset
    data = make_people(tmp_path / "data", ["A", "B", "C", "D"], 15)
    mystery = tmp_path / "mystery"
    mystery.mkdir()
    ds = verify_dataset(data, mystery)
    assert ds["n_identities"] == 4 and ds["n_images"] == 60 and set(ds["counts"].values()) == {15}
    assert ds["duplicates"] == 0 and ds["deterministic"] and not ds["mystery_in_training"]
    shutil.copy(data / "A" / "01.jpg", mystery / "M.jpg")
    shutil.copy(data / "B" / "02.png", data / "B" / "16.png")
    ds = verify_dataset(data, mystery)
    assert ds["mystery_in_training"] == ["M.jpg"] and ds["duplicates"] == 1 and ds["counts"]["B"] == 16


# ---------- diagnostics (same fitted PCA, no new method) ----------

def test_truncate_model_equals_training_with_smaller_k(tmp_path):
    make_people(tmp_path, ["Alice", "Bob", "Cara"], 5)
    split = prepare_custom_split(tmp_path, n_test_per_person=1, seed=0)
    full = train_model(split["X_train"], split["labels_train"], split["train_paths"], k=11)
    small = train_model(split["X_train"], split["labels_train"], split["train_paths"], k=4)
    cut = truncate_model(full, 4)
    assert np.allclose(cut["U_k"], small["U_k"]) and np.allclose(cut["Z_train"], small["Z_train"])
    assert np.isclose(cut["explained_variance"], small["explained_variance"])
    with pytest.raises(ValueError):
        truncate_model(full, 12)


def test_class_mean_projection_is_mean_of_projections(tmp_path):
    """U_k^T (mu_c - mu) = average of z_i over identity c (linearity of the projection)."""
    make_people(tmp_path, ["Alice", "Bob", "Cara"], 5)
    split = prepare_custom_split(tmp_path, n_test_per_person=1, seed=0)
    model = train_model(split["X_train"], split["labels_train"], split["train_paths"], k=6)
    ids, M = class_mean_projections(split["X_train"], split["labels_train"], model)
    assert ids == ["Alice", "Bob", "Cara"] and M.shape == (6, 3)
    for c, name in enumerate(ids):
        assert np.allclose(M[:, c], model["Z_train"][:, split["labels_train"] == name].mean(axis=1))
    z = M[:, 1] + 1e-3
    predicted, distances = nearest_class_mean(z, M, ids)
    assert predicted == "Bob" and np.isclose(distances["Alice"], np.linalg.norm(z - M[:, 0]))


def test_face_space_residual_bounds(tmp_path):
    make_people(tmp_path, ["Alice", "Bob"], 4)
    split = prepare_custom_split(tmp_path, n_test_per_person=0, seed=0)
    model = train_model(split["X_train"], split["labels_train"], split["train_paths"], k=30)  # k = rank
    assert face_space_residual(split["X_train"][:, 0], model) < 1e-8      # training face lies in the span
    other = np.random.default_rng(1).random(split["X_train"].shape[0])
    assert 0.0 < face_space_residual(other, model) <= 1.0


def test_leave_one_out_counts(tmp_path):
    make_people(tmp_path, ["Alice", "Bob", "Cara"], 4)
    X, labels, _, _ = load_labelled_folder(tmp_path)
    correct = leave_one_out(X, labels, [2, 5])
    assert set(correct) == {2, 5} and correct[5] == 12
