"""
Phase 4 tests: projection of new faces, Euclidean distances, nearest-neighbour
recognition, accuracy, confusion matrix, and the absence of test-data leakage.

No test asserts a particular accuracy on the real faces; that is measured by the experiment.
"""

import numpy as np
import pytest

from src.data_loader import DATASET_DIR, load_dataset, train_test_split
from src.pca import center_data, compute_mean_face, compute_svd, get_principal_components
from src.projection import project_data, project_new_face
from src.recognition import accuracy, confusion_matrix, euclidean_distances, recognize, recognize_faces

ATOL = 1e-10


@pytest.fixture
def model():
    """A small PCA 'model' from random training data: d = 30, N = 10, k = 4."""
    X_train = np.random.default_rng(0).random((30, 10))
    mu = compute_mean_face(X_train)
    X_c = center_data(X_train, mu)
    U, _, _ = compute_svd(X_c)
    U_k = get_principal_components(U, 4)
    return X_train, mu, X_c, U, U_k


def fit_and_recognize(X_train, labels_train, X_test, k):
    """The full Phase 4 pipeline, as in experiments/run_recognition.py."""
    mu = compute_mean_face(X_train)                    # training data only
    X_c = center_data(X_train, mu)
    U, _, _ = compute_svd(X_c)                         # training data only
    U_k = get_principal_components(U, k)
    Z_train = project_data(X_c, U_k)
    return mu, U, recognize_faces(X_test, mu, U_k, Z_train, labels_train)


# ---------- projection of a new face ----------

def test_project_new_face_matches_equation(model):
    _, mu, _, _, U_k = model
    x = np.random.default_rng(1).random(30)
    z = project_new_face(x, mu, U_k)
    assert z.shape == (4,)
    assert np.allclose(z, U_k.T @ (x - mu[:, 0]), atol=ATOL)
    assert np.isclose(z[2], U_k[:, 2] @ (x - mu[:, 0]))  # z_j = u_j^T (x - mu)


def test_project_new_face_is_deterministic(model):
    _, mu, _, _, U_k = model
    x = np.random.default_rng(1).random(30)
    assert np.array_equal(project_new_face(x, mu, U_k), project_new_face(x, mu, U_k))


def test_project_new_face_equals_column_of_training_projection(model):
    X_train, mu, X_c, _, U_k = model
    Z_train = project_data(X_c, U_k)
    for i in range(X_train.shape[1]):
        assert np.allclose(project_new_face(X_train[:, i], mu, U_k), Z_train[:, i], atol=ATOL)


def test_project_new_face_rejects_wrong_shapes(model):
    _, mu, _, _, U_k = model
    x = np.random.default_rng(1).random(30)
    with pytest.raises(ValueError):
        project_new_face(x.reshape(30, 1), mu, U_k)    # (d, 1) is not silently flattened
    with pytest.raises(ValueError):
        project_new_face(x[:20], mu, U_k)              # wrong d
    with pytest.raises(ValueError):
        project_new_face(x, mu.ravel(), U_k)           # mean must be (d, 1)


# ---------- training projections ----------

def test_training_projection_shape_and_equation(model):
    X_train, mu, X_c, _, U_k = model
    Z_train = project_data(X_c, U_k)
    assert Z_train.shape == (4, 10)                    # k coordinates per face, one face per column
    assert np.allclose(Z_train, U_k.T @ (X_train - mu), atol=ATOL)
    assert np.array_equal(Z_train, project_data(X_c, U_k))


def test_full_rank_projection_preserves_distances(model):
    """With all rank(X_c) = N - 1 components, U_k is an isometry on the span of the data."""
    X_train, mu, X_c, U, _ = model
    U_full = get_principal_components(U, 9)
    Z = project_data(X_c, U_full)
    for i, j in [(0, 1), (2, 7), (4, 9)]:
        assert np.isclose(np.linalg.norm(Z[:, i] - Z[:, j]), np.linalg.norm(X_train[:, i] - X_train[:, j]))


# ---------- distances and nearest neighbour ----------

def test_euclidean_distances_hand_example():
    database = np.array([[0.0, 3.0, 1.0],
                         [0.0, 4.0, 1.0]])         # points (0,0), (3,4), (1,1) as columns
    distances = euclidean_distances(np.array([0.0, 0.0]), database)
    assert distances.shape == (3,)
    assert np.allclose(distances, [0.0, 5.0, np.sqrt(2.0)])


def test_euclidean_distances_rejects_wrong_shapes():
    database = np.zeros((2, 3))
    with pytest.raises(ValueError):
        euclidean_distances(np.zeros(3), database)         # k mismatch
    with pytest.raises(ValueError):
        euclidean_distances(np.zeros((2, 1)), database)    # query must be (k,)
    with pytest.raises(ValueError):
        euclidean_distances(np.zeros(2), database.T)       # database must be (k, N), not (N, k)


def test_recognize_returns_label_index_and_distance():
    Z_train = np.array([[0.0, 10.0, 4.0],
                        [0.0, 0.0, 3.0]])
    labels = np.array(["alice", "bob", "carol"])
    label, index, distance = recognize(np.array([4.0, 2.0]), Z_train, labels)
    assert (label, index) == ("carol", 2)
    assert np.isclose(distance, 1.0)


def test_recognize_tie_takes_smallest_index():
    Z_train = np.array([[-1.0, 1.0]])
    label, index, _ = recognize(np.array([0.0]), Z_train, np.array(["a", "b"]))
    assert (label, index) == ("a", 0)


def test_recognize_rejects_label_mismatch():
    with pytest.raises(ValueError):
        recognize(np.zeros(2), np.zeros((2, 3)), np.array(["a", "b"]))


def test_recognize_faces_matches_one_by_one(model):
    X_train, mu, X_c, _, U_k = model
    labels = np.array([f"p{i // 2}" for i in range(10)])
    Z_train = project_data(X_c, U_k)
    X_test = np.random.default_rng(3).random((30, 5))
    predictions, indices, distances = recognize_faces(X_test, mu, U_k, Z_train, labels)
    assert predictions.shape == indices.shape == distances.shape == (5,)
    for j in range(5):
        expected = recognize(project_new_face(X_test[:, j], mu, U_k), Z_train, labels)
        assert (predictions[j], indices[j]) == expected[:2]
        assert np.isclose(distances[j], expected[2])
        assert predictions[j] == labels[indices[j]]


# ---------- accuracy and confusion matrix ----------

def test_accuracy_hand_example():
    assert accuracy(np.array(["a", "b", "c", "a"]), np.array(["a", "b", "a", "a"])) == 0.75
    with pytest.raises(ValueError):
        accuracy(np.array(["a"]), np.array(["a", "b"]))
    with pytest.raises(ValueError):
        accuracy(np.array([]), np.array([]))


def test_confusion_matrix_hand_example():
    true = np.array(["a", "a", "b", "c"])
    predicted = np.array(["a", "b", "b", "c"])
    counts = confusion_matrix(true, predicted, ["a", "b", "c"])
    assert np.array_equal(counts, [[1, 1, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]])
    assert np.trace(counts) / counts.sum() == accuracy(predicted, true)


# ---------- no test-data leakage ----------

def test_pca_uses_training_data_only():
    rng = np.random.default_rng(4)
    X_train, X_test = rng.random((30, 12)), rng.random((30, 6)) + 5.0  # very different test data
    labels_train = np.array([f"p{i // 3}" for i in range(12)])
    mu, U, _ = fit_and_recognize(X_train, labels_train, X_test, k=3)
    assert np.allclose(mu[:, 0], X_train.mean(axis=1))                                  # training mean
    assert not np.allclose(mu[:, 0], np.hstack([X_train, X_test]).mean(axis=1))         # not train+test
    U_ref, _, _ = compute_svd(X_train - X_train.mean(axis=1, keepdims=True))
    assert np.allclose(np.abs(U[:, :3]), np.abs(U_ref[:, :3]), atol=1e-8)               # training SVD


def test_each_test_face_is_recognised_independently():
    """If PCA (mu or U) used the test set, changing one test face would change the others' results."""
    rng = np.random.default_rng(5)
    X_train, X_test = rng.random((30, 12)), rng.random((30, 6))
    labels_train = np.array([f"p{i // 3}" for i in range(12)])
    _, _, (pred_a, idx_a, dist_a) = fit_and_recognize(X_train, labels_train, X_test, k=4)
    X_test_changed = X_test.copy()
    X_test_changed[:, 0] = 100.0
    _, _, (pred_b, idx_b, dist_b) = fit_and_recognize(X_train, labels_train, X_test_changed, k=4)
    assert np.array_equal(pred_a[1:], pred_b[1:])
    assert np.array_equal(idx_a[1:], idx_b[1:])
    assert np.array_equal(dist_a[1:], dist_b[1:])


# ---------- small synthetic recognition example ----------

def test_synthetic_recognition_end_to_end():
    """3 'people' = 3 distinct prototype vectors + small noise; new noisy images must be recognised."""
    rng = np.random.default_rng(6)
    d, people = 50, ["anna", "ben", "cara"]
    prototypes = {p: rng.random(d) for p in people}
    make = lambda p: np.clip(prototypes[p] + 0.03 * rng.standard_normal(d), 0, 1)
    X_train = np.column_stack([make(p) for p in people for _ in range(4)])
    labels_train = np.array([p for p in people for _ in range(4)])
    X_test = np.column_stack([make(p) for p in people for _ in range(2)])
    labels_test = np.array([p for p in people for _ in range(2)])
    _, _, (predictions, indices, _) = fit_and_recognize(X_train, labels_train, X_test, k=2)
    assert accuracy(predictions, labels_test) == 1.0
    assert np.all(labels_train[indices] == labels_test)


# ---------- real data shapes (no accuracy assumed) ----------

@pytest.mark.skipif(not DATASET_DIR.is_dir(), reason="AT&T dataset not downloaded")
def test_real_data_projection_shapes():
    X, labels = load_dataset(DATASET_DIR)
    X_train, y_train, X_test, _ = train_test_split(X, labels, n_test_per_person=3, seed=42)
    mu = compute_mean_face(X_train)
    X_c = center_data(X_train, mu)
    U, _, _ = compute_svd(X_c)
    U_k = get_principal_components(U, 20)
    Z_train = project_data(X_c, U_k)
    assert Z_train.shape == (20, 280)
    assert project_new_face(X_test[:, 0], mu, U_k).shape == (20,)
    predictions, _, distances = recognize_faces(X_test, mu, U_k, Z_train, y_train)
    assert predictions.shape == distances.shape == (120,)
    assert set(predictions) <= set(y_train)
