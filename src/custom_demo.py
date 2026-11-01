"""
Custom-dataset demonstration: train the SAME eigenfaces pipeline on your own photos.

Folder layout (folder name = identity label; any number of people):

    demo_data/
        Alice/  01.jpg 02.jpg ...
        Bob/    01.jpg ...

Pipeline (identical to the AT&T experiment, using the same functions):

    image --preprocess_image--> x in R^d
    X_train --compute_mean_face--> mu,  --center_data--> X_c,  --compute_svd--> U, Sigma, V^T
    U_k = U[:, :k],  Z_train = U_k^T X_c
    new image -> z = U_k^T (x - mu) -> d_i = ||z - z_i||_2 -> identity of argmin_i d_i

Nothing here is a new algorithm: this module only organises files, the train/test
split, and saving/loading the trained quantities (mu, U_k, Z_train, labels).
"""

import hashlib
from pathlib import Path

import numpy as np

from src.data_loader import discover_images, load_image, split_indices
from src.pca import (
    center_data, compute_mean_face, compute_svd, explained_variance_ratio,
    get_principal_components, numerical_rank,
)
from src.preprocessing import IMAGE_SHAPE, build_data_matrix, preprocess_image
from src.projection import project_data, project_new_face
from src.recognition import accuracy, euclidean_distances, recognize, recognize_faces

DEFAULT_K = 30
ASPECT_TOLERANCE = 0.10  # note if an image's height/width differs from the target by more than 10 %


# ---------------------------------------------------------------------------
# Loading and splitting
# ---------------------------------------------------------------------------

def file_fingerprint(path: Path) -> str:
    """SHA-256 of the file bytes: identical files have identical fingerprints."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_labelled_folder(
    data_dir: Path,
    image_shape: tuple[int, int] = IMAGE_SHAPE,
) -> tuple[np.ndarray, np.ndarray, list[Path], list[str]]:
    """
    Load every image under data_dir/<identity>/ with the project preprocessing.

    Output: X (d, N) with one face per column, labels (N,), paths (N), and warnings
    for images whose aspect ratio differs strongly from image_shape (those images are
    centre-cropped before resizing, so part of the border is discarded: check that the
    face is centred).
    """
    paths, labels = discover_images(Path(data_dir))
    target_ratio = image_shape[0] / image_shape[1]
    vectors, warnings = [], []
    for path in paths:
        image = load_image(path)
        ratio = image.height / image.width
        if abs(ratio - target_ratio) / target_ratio > ASPECT_TOLERANCE:
            warnings.append(f"{path.parent.name}/{path.name}: {image.width} x {image.height} (w x h), "
                            f"height/width = {ratio:.2f}, target {target_ratio:.2f} -> centre-cropped (not stretched)")
        vectors.append(preprocess_image(image, image_shape))
    return build_data_matrix(vectors), np.array(labels), paths, warnings


def check_no_overlap(train_paths: list[Path], test_paths: list[Path]) -> None:
    """
    Raise ValueError if any test file is byte-identical to a training file.

    Evaluating on an image that was used for training is not a recognition test:
    its distance to itself in PCA space is exactly 0.
    """
    train_prints = {file_fingerprint(p): p for p in train_paths}
    duplicates = [(t, train_prints[file_fingerprint(t)]) for t in test_paths if file_fingerprint(t) in train_prints]
    if duplicates:
        listing = "\n".join(f"  test {t}  ==  train {s}" for t, s in duplicates)
        raise ValueError(f"Test images identical to training images (not allowed):\n{listing}")


def prepare_custom_split(
    data_dir: Path,
    image_shape: tuple[int, int] = IMAGE_SHAPE,
    n_test_per_person: int = 3,
    seed: int = 42,
    test_dir: Path | None = None,
) -> dict:
    """
    Build the training and test sets for the custom demo.

    Two modes:
      - test_dir is None: per person, n_test_per_person images of data_dir are held out at
        random (fixed seed), using the same split_indices as the AT&T experiment.
      - test_dir given: ALL images of data_dir are training images, and the images in
        test_dir/<identity>/ are the unseen test images.
    In both modes no test file may be byte-identical to a training file.
    """
    X, labels, paths, warnings = load_labelled_folder(data_dir, image_shape)
    if test_dir is None:
        is_test = split_indices(labels, n_test_per_person, seed)
        split = {
            "X_train": X[:, ~is_test], "labels_train": labels[~is_test],
            "train_paths": [p for p, t in zip(paths, is_test) if not t],
            "X_test": X[:, is_test], "labels_test": labels[is_test],
            "test_paths": [p for p, t in zip(paths, is_test) if t],
        }
    else:
        X_test, labels_test, test_paths, test_warnings = load_labelled_folder(test_dir, image_shape)
        warnings += test_warnings
        split = {"X_train": X, "labels_train": labels, "train_paths": paths,
                 "X_test": X_test, "labels_test": labels_test, "test_paths": test_paths}
    check_no_overlap(split["train_paths"], split["test_paths"])
    split["warnings"] = warnings
    return split


# ---------------------------------------------------------------------------
# Training (the existing PCA) and recognition (the existing nearest neighbour)
# ---------------------------------------------------------------------------

def train_model(
    X_train: np.ndarray,
    labels_train: np.ndarray,
    train_paths: list[Path],
    k: int = DEFAULT_K,
    image_shape: tuple[int, int] = IMAGE_SHAPE,
) -> dict:
    """
    Eigenfaces training with the project's PCA functions:
        mu = (1/N) X 1,  X_c = X - mu,  X_c = U Sigma V^T,  U_k = U[:, :k],  Z_train = U_k^T X_c.

    k cannot exceed rank(X_c) <= N_train - 1: directions beyond the rank have sigma = 0 and
    carry no information. A larger requested k is reduced to the rank (reported in the result).
    """
    if X_train.shape[1] < 2:
        raise ValueError("Need at least 2 training images")
    mu = compute_mean_face(X_train)
    X_c = center_data(X_train, mu)
    U, S, _ = compute_svd(X_c)
    rank = numerical_rank(S, X_c.shape)
    if rank == 0:
        raise ValueError("All training images are identical after preprocessing: no variation for PCA")
    k_used = min(k, rank)
    U_k = get_principal_components(U, k_used)
    _, cumulative = explained_variance_ratio(S)
    return {
        "mean_face": mu,                                    # (d, 1)
        "U_k": U_k,                                         # (d, k)
        "Z_train": project_data(X_c, U_k),                  # (k, N_train)
        "labels_train": np.asarray(labels_train),           # (N_train,)
        "train_files": np.array([str(p) for p in train_paths]),
        "singular_values": S,                               # (r,)
        "image_shape": (int(image_shape[0]), int(image_shape[1])),
        "k_requested": k,
        "k": k_used,
        "rank": rank,
        "explained_variance": float(cumulative[k_used - 1]),
    }


def save_model(model: dict, path: Path) -> None:
    """Store the trained quantities locally as a NumPy .npz file (no pickling)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in model.items()})


def load_model(path: Path) -> dict:
    """Inverse of save_model."""
    with np.load(path, allow_pickle=False) as data:
        model = {key: data[key] for key in data.files}
    for key in ("k_requested", "k", "rank"):
        model[key] = int(model[key])
    model["explained_variance"] = float(model["explained_variance"])
    model["image_shape"] = tuple(int(v) for v in model["image_shape"])
    return model


def recognize_image(model: dict, image_path: Path) -> dict:
    """
    Recognise one image file with a trained model.

        x = preprocess(image),  z = U_k^T (x - mu),  d_i = ||z - z_i||_2,  i* = argmin_i d_i

    Returns the prediction, the nearest training file, its distance, the smallest distance
    to each identity (a distance, NOT a probability), and whether the file is byte-identical
    to a training image (then the result is not a genuine recognition test).
    """
    image_shape = (int(model["image_shape"][0]), int(model["image_shape"][1]))
    x = preprocess_image(load_image(Path(image_path)), image_shape)
    z = project_new_face(x, model["mean_face"], model["U_k"])
    distances = euclidean_distances(z, model["Z_train"])
    label, nearest, distance = recognize(z, model["Z_train"], model["labels_train"])
    labels = model["labels_train"]
    identities = sorted(set(labels.tolist()), key=lambda s: s.lower())
    per_identity = {person: float(distances[labels == person].min()) for person in identities}
    fingerprint = file_fingerprint(Path(image_path))
    in_training = any(Path(f).exists() and file_fingerprint(Path(f)) == fingerprint for f in model["train_files"])
    return {
        "x": x, "z": z, "distances": distances,
        "predicted": label, "nearest_index": nearest, "nearest_file": str(model["train_files"][nearest]),
        "distance": distance, "per_identity": per_identity, "in_training_set": in_training,
    }


def truncate_model(model: dict, k: int) -> dict:
    """
    The same fitted PCA with fewer components:  U_k = U[:, :k],  Z_train = (first k rows of Z_train).

    No refitting: mu, U and Sigma are unchanged. Row j of Z_train is u_j^T X_c, so keeping the
    first k rows gives exactly U_k^T X_c. Requires 1 <= k <= model["k"].
    """
    if not 1 <= k <= model["k"]:
        raise ValueError(f"k must be between 1 and {model['k']}, got {k}")
    _, cumulative = explained_variance_ratio(model["singular_values"])
    return {**model, "U_k": model["U_k"][:, :k], "Z_train": model["Z_train"][:k],
            "k": k, "explained_variance": float(cumulative[k - 1])}


def class_mean_projections(X_train: np.ndarray, labels_train: np.ndarray, model: dict) -> tuple[list[str], np.ndarray]:
    """
    DIAGNOSTIC only (not the official recognition rule): project each identity's mean face.

        mu_c = mean of the training columns of identity c   (in R^d)
        m_c  = U_k^T (mu_c - mu)                             (in R^k)

    Since projection is linear, m_c is also the average of that identity's z_i.
    Output: (identities, M) with M of shape (k, C), column c = m_c.
    """
    labels_train = np.asarray(labels_train)
    identities = sorted(set(labels_train.tolist()), key=str.lower)
    class_means = np.column_stack([X_train[:, labels_train == c].mean(axis=1) for c in identities])  # (d, C)
    return identities, project_data(class_means - model["mean_face"], model["U_k"])


def nearest_class_mean(z: np.ndarray, class_projections: np.ndarray, identities: list[str]) -> tuple[str, dict]:
    """
    DIAGNOSTIC nearest-class-mean rule:  d_c = ||z - m_c||_2,  prediction = argmin_c d_c.
    Returns (predicted identity, {identity: d_c}).
    """
    distances = euclidean_distances(z, class_projections)
    return identities[int(np.argmin(distances))], {c: float(v) for c, v in zip(identities, distances)}


def face_space_residual(x: np.ndarray, model: dict) -> float:
    """
    Distance from the face space, relative:  ||(x - mu) - U_k U_k^T (x - mu)|| / ||x - mu||.

    0 = the centered image lies in span(U_k); 1 = it is orthogonal to every eigenface.
    A large value means the image does not look like the training images (e.g. different framing).
    """
    x_c = x - model["mean_face"][:, 0]
    z = model["U_k"].T @ x_c
    return float(np.linalg.norm(x_c - model["U_k"] @ z) / np.linalg.norm(x_c))


def leave_one_out(X: np.ndarray, labels: np.ndarray, ks: list[int], image_shape: tuple[int, int] = IMAGE_SHAPE) -> dict:
    """
    Leave-one-out nearest-neighbour accuracy: for each column j, PCA is fitted on the other
    N - 1 columns only, and column j is recognised with every k in ks (one SVD per j, shared by all k).

    Output: {k: number of correctly recognised columns}.
    """
    labels = np.asarray(labels)
    correct = {k: 0 for k in ks}
    for j in range(X.shape[1]):
        keep = np.arange(X.shape[1]) != j
        full = train_model(X[:, keep], labels[keep], [""] * int(keep.sum()), max(ks), image_shape)
        for k in ks:
            model = truncate_model(full, min(k, full["k"]))
            predicted, _, _ = recognize_faces(X[:, [j]], model["mean_face"], model["U_k"],
                                              model["Z_train"], model["labels_train"])
            correct[k] += int(predicted[0] == labels[j])
    return correct


def evaluate_split(model: dict, split: dict) -> dict:
    """Recognise every held-out test image; accuracy over identities known to the model."""
    results = [recognize_image(model, p) for p in split["test_paths"]]
    predictions = np.array([r["predicted"] for r in results])
    truth = split["labels_test"]
    known = np.isin(truth, model["labels_train"])
    acc = accuracy(predictions[known], truth[known]) if known.any() else float("nan")
    return {"results": results, "predictions": predictions, "accuracy": acc, "known": known}
