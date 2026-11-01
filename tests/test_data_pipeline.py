"""
Phase 2 tests: image -> vector -> data matrix X (d x N), labels, and the train/test split.

Most tests use small synthetic images written to a temporary folder, so they do
not depend on the downloaded dataset. The last test checks the real AT&T data
and is skipped if it has not been downloaded.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data_loader import DATASET_DIR, discover_images, load_dataset, load_image, train_test_split
from src.preprocessing import (
    IMAGE_SHAPE, build_data_matrix, center_crop_to_aspect, image_to_array, image_to_vector,
    preprocess_image, resize_image, to_grayscale, vector_to_image, verify_data_matrix,
)


@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    """3 people x 4 images, 92 x 112 grayscale PGM, each image a distinct constant shade."""
    for person in (1, 2, 10):  # includes s10 to check numeric (not alphabetical) ordering
        person_dir = tmp_path / f"s{person}"
        person_dir.mkdir()
        for image_number in range(1, 5):
            shade = 10 * person + image_number
            Image.new("L", (92, 112), color=shade).save(person_dir / f"{image_number}.pgm")
    return tmp_path


# ---------- single image ----------

def test_load_image_reads_pgm(fake_dataset):
    image = load_image(fake_dataset / "s1" / "1.pgm")
    assert image.mode == "L"
    assert image.size == (92, 112)  # Pillow reports (width, height)


def test_grayscale_conversion_uses_luma_formula():
    rgb = Image.new("RGB", (4, 4), color=(200, 100, 50))
    gray = to_grayscale(rgb)
    assert gray.mode == "L"
    expected = round(0.299 * 200 + 0.587 * 100 + 0.114 * 50)
    assert abs(int(np.asarray(gray)[0, 0]) - expected) <= 1


def test_grayscale_image_is_unchanged():
    gray = Image.new("L", (5, 5), color=77)
    assert np.array_equal(np.asarray(to_grayscale(gray)), np.asarray(gray))


def test_resize_uses_height_width_order():
    image = Image.new("L", (92, 112))
    assert np.asarray(resize_image(image, (78, 64))).shape == (78, 64)
    # non-square target: (height=32, width=16) must give 32 rows, 16 columns
    assert np.asarray(resize_image(image, (32, 16))).shape == (32, 16)


def test_image_to_array_scales_to_unit_interval():
    pixels = np.array([[0, 51], [204, 255]], dtype=np.uint8)
    array = image_to_array(Image.fromarray(pixels))  # uint8 2-D array -> mode "L"
    assert array.dtype == np.float64
    assert np.allclose(array, pixels / 255.0)
    assert array.min() == 0.0 and array.max() == 1.0


def test_flatten_is_row_major_and_invertible():
    h, w = 3, 4
    image = np.arange(h * w, dtype=float).reshape(h, w)
    x = image_to_vector(image)
    assert x.shape == (h * w,)
    for r in range(h):
        for c in range(w):
            assert x[r * w + c] == image[r, c]
    assert np.array_equal(vector_to_image(x, (h, w)), image)


def test_vector_to_image_rejects_wrong_length():
    with pytest.raises(ValueError):
        vector_to_image(np.zeros(10), (3, 4))


def test_preprocessed_vector_has_dimension_4992():
    x = preprocess_image(Image.new("RGB", (92, 112), color=(10, 20, 30)))  # default IMAGE_SHAPE
    assert IMAGE_SHAPE == (78, 64)
    assert x.shape == (4992,)
    assert 0.0 <= x.min() <= x.max() <= 1.0


# ---------- aspect ratio: centre crop, never stretch ----------

def square_on_white(width: int, height: int, side: int) -> Image.Image:
    """White W x H image with a black side x side square in the centre."""
    pixels = np.full((height, width), 255, dtype=np.uint8)
    top, left = (height - side) // 2, (width - side) // 2
    pixels[top:top + side, left:left + side] = 0
    return Image.fromarray(pixels)


def dark_extent(image_array: np.ndarray) -> tuple[int, int]:
    """(number of rows, number of columns) that contain a dark pixel."""
    dark = image_array < 0.5
    return int(dark.any(axis=1).sum()), int(dark.any(axis=0).sum())


@pytest.mark.parametrize("size", [(448, 672), (672, 672), (672, 448), (92, 112), (500, 1000), (1000, 300)])
def test_output_is_always_78_by_64(size):
    image = Image.new("RGB", size, color=(90, 120, 150))
    x = preprocess_image(image)
    assert x.shape == (4992,)
    assert vector_to_image(x, IMAGE_SHAPE).shape == (78, 64)


@pytest.mark.parametrize("size", [(448, 672), (672, 672), (672, 448), (500, 1000), (1000, 300)])
def test_center_crop_has_target_ratio(size):
    cropped = center_crop_to_aspect(Image.new("L", size), IMAGE_SHAPE)
    W, H = cropped.size
    assert W <= size[0] and H <= size[1]
    assert W == size[0] or H == size[1]                    # largest possible crop
    assert abs(H / W - 78 / 64) < 1.0 / min(W, H)          # ratio exact up to pixel rounding


def test_center_crop_of_custom_photo_is_symmetric():
    """448 x 672 (w x h): keep all 448 columns and the middle 546 rows (63 removed at top and bottom)."""
    pixels = np.arange(672, dtype=np.uint16)[:, None].repeat(448, axis=1).astype(np.int32)
    cropped = np.asarray(center_crop_to_aspect(Image.fromarray(pixels), IMAGE_SHAPE))
    assert cropped.shape == (546, 448)
    assert cropped[0, 0] == 63 and cropped[-1, 0] == 608   # 63 rows above, 672 - 609 = 63 rows below


@pytest.mark.parametrize("size", [(448, 672), (672, 672), (672, 448)])
def test_square_stays_square(size):
    """A black square in a portrait, square or landscape photo must still be square after preprocessing."""
    x = preprocess_image(square_on_white(*size, side=200))
    rows, cols = dark_extent(vector_to_image(x, IMAGE_SHAPE))
    assert rows > 10 and abs(rows - cols) <= 1


def test_direct_resize_would_have_stretched_the_square():
    """The old method (resize 448 x 672 straight to 64 x 78) squashes the square; the test above would catch it."""
    stretched = image_to_array(resize_image(square_on_white(448, 672, side=200), IMAGE_SHAPE))
    rows, cols = dark_extent(stretched)
    assert cols - rows >= 4                                # about 23 rows vs 29 columns


def test_att_images_are_not_cropped():
    """112 x 92 (h x w) already has ratio 78/64 to within half a pixel: AT&T preprocessing is unchanged."""
    rng = np.random.default_rng(0)
    image = Image.fromarray((rng.random((112, 92)) * 255).astype(np.uint8))
    assert center_crop_to_aspect(image, IMAGE_SHAPE).size == (92, 112)
    direct = image_to_vector(image_to_array(resize_image(image, IMAGE_SHAPE)))
    assert np.array_equal(preprocess_image(image), direct)


def test_normalization_after_crop():
    for shade in (0, 51, 128, 255):
        x = preprocess_image(Image.new("L", (448, 672), color=shade))
        assert x.dtype == np.float64
        assert np.allclose(x, shade / 255.0)
    x = preprocess_image(square_on_white(448, 672, side=200))
    assert x.min() == 0.0 and x.max() == 1.0


# ---------- data matrix ----------

def test_build_data_matrix_puts_vectors_in_columns():
    vectors = [np.full(6, float(j)) for j in range(5)]
    X = build_data_matrix(vectors)
    assert X.shape == (6, 5)
    for j, v in enumerate(vectors):
        assert np.array_equal(X[:, j], v)


def test_discover_images_numeric_order(fake_dataset):
    paths, labels = discover_images(fake_dataset)
    assert labels == ["s1"] * 4 + ["s2"] * 4 + ["s10"] * 4
    assert [p.name for p in paths[:4]] == ["1.pgm", "2.pgm", "3.pgm", "4.pgm"]


def test_load_dataset_shapes_and_label_consistency(fake_dataset):
    X, labels = load_dataset(fake_dataset, (78, 64))
    assert X.shape == (4992, 12)
    assert labels.shape == (12,)
    assert X.shape[1] == len(labels)
    # column j must be the image whose label is labels[j]: shades were 10*person + image_number
    paths, _ = discover_images(fake_dataset)
    for j, path in enumerate(paths):
        assert labels[j] == path.parent.name
        expected_shade = (10 * int(path.parent.name[1:]) + int(path.stem)) / 255.0
        assert np.allclose(X[:, j], expected_shade)


def test_verify_data_matrix_reports_and_rejects(fake_dataset):
    X, labels = load_dataset(fake_dataset, (78, 64))
    summary = verify_data_matrix(X, labels, (78, 64))
    assert summary["data_matrix_shape"] == (4992, 12)
    assert summary["labels_shape"] == (12,)
    assert summary["n_people"] == 3
    with pytest.raises(ValueError):
        verify_data_matrix(X, labels[:-1], (78, 64))  # label count mismatch
    with pytest.raises(ValueError):
        verify_data_matrix(X, labels, (32, 32))       # wrong d


# ---------- train / test split ----------

def test_split_is_deterministic(fake_dataset):
    X, labels = load_dataset(fake_dataset, (8, 8))
    a = train_test_split(X, labels, n_test_per_person=1, seed=0)
    b = train_test_split(X, labels, n_test_per_person=1, seed=0)
    for part_a, part_b in zip(a, b):
        assert np.array_equal(part_a, part_b)


def test_split_partitions_columns_per_person(fake_dataset):
    X, labels = load_dataset(fake_dataset, (8, 8))
    X_train, y_train, X_test, y_test = train_test_split(X, labels, n_test_per_person=1, seed=0)
    assert X_train.shape == (64, 9) and X_test.shape == (64, 3)
    assert X_train.shape[1] == len(y_train) and X_test.shape[1] == len(y_test)
    # every person is in both sets, with the requested counts
    for person in np.unique(labels):
        assert np.sum(y_train == person) == 3
        assert np.sum(y_test == person) == 1
    # train and test are disjoint and together give back all columns of X
    all_columns = np.concatenate([X_train, X_test], axis=1)
    original = {tuple(col) for col in X.T}
    assert {tuple(col) for col in all_columns.T} == original
    assert len(original) == 12


def test_split_rejects_too_many_test_images(fake_dataset):
    X, labels = load_dataset(fake_dataset, (8, 8))
    with pytest.raises(ValueError):
        train_test_split(X, labels, n_test_per_person=4, seed=0)


# ---------- real dataset (optional) ----------

@pytest.mark.skipif(not DATASET_DIR.is_dir(), reason="AT&T dataset not downloaded")
def test_real_dataset_matrix():
    X, labels = load_dataset(DATASET_DIR, (78, 64))
    assert X.shape == (4992, 400)
    assert labels.shape == (400,)
    assert len(np.unique(labels)) == 40
    X_train, y_train, X_test, y_test = train_test_split(X, labels, n_test_per_person=3, seed=42)
    assert X_train.shape == (4992, 280) and X_test.shape == (4992, 120)
