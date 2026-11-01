"""
Obtaining the face dataset and assembling the data matrix X with its labels.

Dataset: AT&T (formerly ORL) Database of Faces, Olivetti Research Laboratory,
Cambridge, 1992-1994. 40 people x 10 images, 92 x 112 pixels, 8-bit grayscale.
Folder layout after extraction:

    data/raw/att_faces/s1/1.pgm ... s1/10.pgm
    ...
    data/raw/att_faces/s40/1.pgm ... s40/10.pgm

Matrix convention: X has shape (d, N), one face per COLUMN.
labels has shape (N,), and labels[j] is the person shown in column j of X.
Identity is kept OUT of X; X contains pixel intensities only.
"""

import hashlib
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.preprocessing import IMAGE_SHAPE, build_data_matrix, preprocess_image

DATASET_URL = "https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip"
DATASET_SHA256 = "509d763bd2276aa054d260d7f1000e8be212738134f2f83d8ff13bc2adf1c8ec"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_DIR = RAW_DIR / "att_faces"

IMAGE_EXTENSIONS = {".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def download_att_faces(raw_dir: Path = RAW_DIR) -> Path:
    """
    Download (about 3.8 MB) and extract the AT&T face database, if not already present.

    Output: path to the extracted folder (raw_dir / "att_faces").
    No authentication is required. Existing files are never re-downloaded.
    Before extraction the zip's SHA-256 is compared with DATASET_SHA256, so every
    user works with exactly the same 400 images.
    """
    dataset_dir = raw_dir / "att_faces"
    if dataset_dir.is_dir() and any(dataset_dir.glob("s*/*.pgm")):
        return dataset_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "att_faces.zip"
    if not zip_path.exists():
        print(f"Downloading {DATASET_URL} ...")
        urllib.request.urlretrieve(DATASET_URL, zip_path)

    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if digest != DATASET_SHA256:
        raise ValueError(f"{zip_path} has SHA-256 {digest}, expected {DATASET_SHA256}. "
                         "Delete it and download again.")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dataset_dir)
    return dataset_dir


def _natural_key(path: Path) -> tuple:
    """
    "Natural" sort key: digit runs compare as numbers, other text alphabetically
    (case-insensitive). So s2 < s10, 2.pgm < 10.pgm, and Alice < Bob.
    Every name gets a distinct, platform-independent position.
    """
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts), path.name


def discover_images(dataset_dir: Path) -> tuple[list[Path], list[str]]:
    """
    Find every image file in a "one sub-folder per person" layout.

    Input:  dataset_dir containing sub-folders (one per person) of image files.
    Output: (paths, labels), both of length N; labels[j] is the folder name of paths[j].

    Order is deterministic on every operating system: people in natural order
    (s1, s2, ..., s40 or Alice, Bob, ...), then images in natural order (1, 2, ..., 10).
    This fixes the column order of X. Hidden folders (".name") are ignored.
    """
    person_dirs = sorted(
        (p for p in dataset_dir.iterdir() if p.is_dir() and not p.name.startswith(".")),
        key=_natural_key,
    )
    paths: list[Path] = []
    labels: list[str] = []
    for person_dir in person_dirs:
        image_files = sorted(
            (f for f in person_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS),
            key=_natural_key,
        )
        for image_file in image_files:
            paths.append(image_file)
            labels.append(person_dir.name)
    if not paths:
        raise FileNotFoundError(f"No images found under {dataset_dir}")
    return paths, labels


def load_image(path: Path) -> Image.Image:
    """
    Read one image file from disk, upright.

    Input:  path to an image file.
    Output: Pillow image (for AT&T: mode "L", size 92 x 112).

    Phone cameras often store a photo sideways plus an EXIF "orientation" tag;
    ImageOps.exif_transpose applies that rotation so the face is upright. Files
    without the tag (such as the AT&T PGM images) are returned unchanged.
    """
    with Image.open(path) as image:
        image.load()
        return ImageOps.exif_transpose(image).copy()


def load_dataset(
    dataset_dir: Path = DATASET_DIR,
    image_shape: tuple[int, int] = IMAGE_SHAPE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load every face, preprocess it to a vector, and assemble X column by column.

    Input:  dataset_dir (one sub-folder per person); image_shape = (height, width).
    Output: X of shape (d, N) with d = height * width, values in [0, 1];
            labels of shape (N,), labels[j] = person in column j of X.
    """
    paths, labels = discover_images(dataset_dir)
    vectors = [preprocess_image(load_image(p), image_shape) for p in paths]
    X = build_data_matrix(vectors)
    return X, np.array(labels)


def split_indices(labels: np.ndarray, n_test_per_person: int, seed: int) -> np.ndarray:
    """
    Choose, for each person, n_test_per_person of their columns at random (fixed seed) for testing.

    Input:  labels of shape (N,); n_test_per_person >= 0; seed.
    Output: boolean mask is_test of shape (N,). Every person keeps at least one training image.
    People are visited in sorted label order, so the result depends only on labels and seed.
    """
    if n_test_per_person < 0:
        raise ValueError("n_test_per_person must be >= 0")
    rng = np.random.default_rng(seed)
    test_columns: list[int] = []
    for person in np.unique(labels):
        person_columns = np.flatnonzero(labels == person)
        if len(person_columns) <= n_test_per_person:
            raise ValueError(
                f"Person {person} has {len(person_columns)} images: too few for "
                f"{n_test_per_person} test images plus at least one training image"
            )
        chosen = rng.choice(person_columns, size=n_test_per_person, replace=False)
        test_columns.extend(chosen.tolist())

    is_test = np.zeros(len(labels), dtype=bool)
    is_test[test_columns] = True
    return is_test


def train_test_split(
    X: np.ndarray,
    labels: np.ndarray,
    n_test_per_person: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reproducible per-person split of the COLUMNS of X into training and test sets.

    For each person, n_test_per_person of their images are chosen at random
    (with a fixed seed) for testing; the rest are used for training. Every person
    therefore appears in both sets, so the test asks: "can a NEW image of a KNOWN
    person be recognised?" Train and test columns never overlap.

    Input:  X of shape (d, N), labels of shape (N,), n_test_per_person >= 1, seed.
    Output: X_train (d, N_train), labels_train (N_train,),
            X_test  (d, N_test),  labels_test  (N_test,).
    Columns keep their original relative order inside each set.
    """
    if X.shape[1] != len(labels):
        raise ValueError("X must have one column per label")
    is_test = split_indices(labels, n_test_per_person, seed)
    return X[:, ~is_test], labels[~is_test], X[:, is_test], labels[is_test]
