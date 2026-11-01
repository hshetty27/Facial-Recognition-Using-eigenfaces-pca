"""
Phase 3 tests: mean face, centering, SVD, eigenvalues, explained variance,
principal components, projection and reconstruction.

Convention throughout: X has shape (d, N), one sample per column.
Random test matrices use a fixed seed, so every run is identical.
"""

import numpy as np
import pytest

from src.data_loader import DATASET_DIR, load_dataset, train_test_split
from src.pca import (
    center_data, choose_k, compute_mean_face, compute_svd,
    eigenvalues_from_singular_values, explained_variance_ratio, get_principal_components,
)
from src.projection import project_data, reconstruct, reconstruction_error, relative_reconstruction_error

ATOL = 1e-10


@pytest.fixture
def wide_data() -> np.ndarray:
    """d = 40 'pixels', N = 12 samples (d > N, like the face data)."""
    return np.random.default_rng(0).random((40, 12))


@pytest.fixture
def tall_data() -> np.ndarray:
    """d = 5, N = 30 samples (d < N), small enough to form C explicitly."""
    rng = np.random.default_rng(1)
    return rng.random((5, 30)) * np.array([[5.0], [3.0], [2.0], [1.0], [0.5]])


def pca_parts(X):
    mu = compute_mean_face(X)
    X_c = center_data(X, mu)
    U, S, Vt = compute_svd(X_c)
    return mu, X_c, U, S, Vt


# ---------- 1. mean ----------

def test_mean_hand_example():
    X = np.array([[1.0, 2.0, 6.0],
                  [0.0, 4.0, 8.0]])  # d = 2, N = 3
    mu = compute_mean_face(X)
    assert mu.shape == (2, 1)
    assert np.allclose(mu, [[3.0], [4.0]])  # (1+2+6)/3, (0+4+8)/3


def test_mean_is_across_columns(wide_data):
    mu = compute_mean_face(wide_data)
    assert mu.shape == (40, 1)
    assert np.allclose(mu[:, 0], wide_data.mean(axis=1))  # average of each row = each pixel
    assert not np.allclose(mu[:12, 0], wide_data.mean(axis=0))  # not the per-column average


# ---------- 2. centering ----------

def test_centering_hand_example():
    X = np.array([[1.0, 2.0, 6.0],
                  [0.0, 4.0, 8.0]])
    X_c = center_data(X, compute_mean_face(X))
    assert np.allclose(X_c, [[-2.0, -1.0, 3.0],
                             [-4.0, 0.0, 4.0]])


def test_centered_rows_have_zero_mean(wide_data):
    X_c = center_data(wide_data, compute_mean_face(wide_data))
    assert X_c.shape == wide_data.shape
    assert np.allclose(X_c.mean(axis=1), 0.0, atol=ATOL)       # every pixel has mean 0
    assert np.allclose(X_c @ np.ones(12), 0.0, atol=ATOL)       # i.e. X_c 1 = 0


def test_centering_rejects_wrong_mean_shape(wide_data):
    mu = compute_mean_face(wide_data)
    with pytest.raises(ValueError):
        center_data(wide_data, mu.ravel())  # (d,) instead of (d, 1)
    with pytest.raises(ValueError):
        center_data(wide_data, mu.T)        # (1, d): no silent transposing


# ---------- 3. SVD ----------

def test_svd_shapes(wide_data, tall_data):
    _, _, U, S, Vt = pca_parts(wide_data)       # d = 40, N = 12 -> r = 12
    assert U.shape == (40, 12) and S.shape == (12,) and Vt.shape == (12, 12)
    _, _, U, S, Vt = pca_parts(tall_data)       # d = 5, N = 30 -> r = 5
    assert U.shape == (5, 5) and S.shape == (5,) and Vt.shape == (5, 30)


def test_svd_reproduces_centered_data(wide_data):
    _, X_c, U, S, Vt = pca_parts(wide_data)
    assert np.allclose(U @ np.diag(S) @ Vt, X_c, atol=ATOL)


def test_singular_values_sorted_and_nonnegative(wide_data):
    _, _, _, S, _ = pca_parts(wide_data)
    assert np.all(S >= 0)
    assert np.all(np.diff(S) <= 0)


def test_centering_reduces_rank_to_n_minus_1(wide_data):
    # X_c 1 = 0, so rank(X_c) <= N - 1: with d > N the last singular value vanishes
    _, _, _, S, _ = pca_parts(wide_data)
    assert S[-1] < 1e-10 * S[0]
    assert S[-2] > 1e-6 * S[0]


# ---------- 4. orthonormality ----------

def test_U_and_V_are_orthonormal(wide_data):
    _, _, U, _, Vt = pca_parts(wide_data)
    assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=ATOL)
    assert np.allclose(Vt @ Vt.T, np.eye(Vt.shape[0]), atol=ATOL)


def test_principal_components_orthonormal(wide_data):
    _, _, U, _, _ = pca_parts(wide_data)
    U_k = get_principal_components(U, 4)
    assert U_k.shape == (40, 4)
    assert np.allclose(U_k, U[:, :4])
    assert np.allclose(U_k.T @ U_k, np.eye(4), atol=ATOL)
    assert abs(U_k[:, 0] @ U_k[:, 1]) < ATOL  # u_1 is orthogonal to u_2


def test_get_principal_components_rejects_invalid_k(wide_data):
    _, _, U, _, _ = pca_parts(wide_data)
    for bad_k in (0, U.shape[1] + 1):
        with pytest.raises(ValueError):
            get_principal_components(U, bad_k)


# ---------- 5. eigenvalue relationship ----------

def test_eigenvalue_formula():
    S = np.array([6.0, 3.0, 0.0])
    assert np.allclose(eigenvalues_from_singular_values(S, n_samples=4), [12.0, 3.0, 0.0])  # sigma^2 / 3
    with pytest.raises(ValueError):
        eigenvalues_from_singular_values(S, n_samples=1)


def test_columns_of_U_are_eigenvectors_of_covariance(tall_data):
    """Form C explicitly (only possible because d = 5 is small) and check C u_i = lambda_i u_i."""
    _, X_c, U, S, _ = pca_parts(tall_data)
    N = tall_data.shape[1]
    C = X_c @ X_c.T / (N - 1)
    assert np.allclose(C, np.cov(tall_data), atol=ATOL)  # np.cov treats rows as variables: same convention
    lam = eigenvalues_from_singular_values(S, N)
    for i in range(U.shape[1]):
        assert np.allclose(C @ U[:, i], lam[i] * U[:, i], atol=ATOL)
    assert np.allclose(C, U @ np.diag(lam) @ U.T, atol=ATOL)  # C = U diag(lambda) U^T


def test_eigenvalues_match_independent_eigensolver(tall_data):
    """Cross-check only: numpy.linalg.eigvalsh is NOT used by the PCA code itself."""
    _, X_c, _, S, _ = pca_parts(tall_data)
    N = tall_data.shape[1]
    C = X_c @ X_c.T / (N - 1)
    reference = np.sort(np.linalg.eigvalsh(C))[::-1]
    assert np.allclose(eigenvalues_from_singular_values(S, N), reference, atol=ATOL)


def test_total_variance_equals_trace_of_covariance(tall_data):
    _, X_c, _, S, _ = pca_parts(tall_data)
    N = tall_data.shape[1]
    total_variance = np.sum(np.var(tall_data, axis=1, ddof=1))  # sum of per-pixel variances
    assert np.isclose(eigenvalues_from_singular_values(S, N).sum(), total_variance)


def test_first_direction_has_maximum_variance(wide_data):
    """Variance of the data along u_1 is lambda_1, and no random unit direction does better."""
    _, X_c, U, S, _ = pca_parts(wide_data)
    N = wide_data.shape[1]
    lam = eigenvalues_from_singular_values(S, N)
    variance_along = lambda u: np.sum((u @ X_c) ** 2) / (N - 1)
    assert np.isclose(variance_along(U[:, 0]), lam[0])
    rng = np.random.default_rng(2)
    for _ in range(200):
        u = rng.standard_normal(40)
        u /= np.linalg.norm(u)
        assert variance_along(u) <= lam[0] + ATOL


# ---------- 6. explained variance and choice of k ----------

def test_explained_variance_hand_example():
    evr, cumulative = explained_variance_ratio(np.array([3.0, 2.0, 1.0, 1.0]))  # squares 9, 4, 1, 1
    assert np.allclose(evr, [9 / 15, 4 / 15, 1 / 15, 1 / 15])
    assert np.allclose(cumulative, [9 / 15, 13 / 15, 14 / 15, 1.0])


def test_explained_variance_sums_to_one_and_is_monotone(wide_data):
    _, _, _, S, _ = pca_parts(wide_data)
    evr, cumulative = explained_variance_ratio(S)
    assert np.isclose(evr.sum(), 1.0)
    assert np.all(evr >= 0)
    assert np.all(np.diff(cumulative) >= -ATOL)
    assert np.isclose(cumulative[-1], 1.0)


def test_choose_k_hand_example():
    cumulative = np.array([0.5, 0.8, 0.95, 1.0])
    assert choose_k(cumulative, 0.5) == 1
    assert choose_k(cumulative, 0.8) == 2
    assert choose_k(cumulative, 0.81) == 3
    assert choose_k(cumulative, 0.90) == 3
    assert choose_k(cumulative, 1.0) == 4
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError):
            choose_k(cumulative, bad)


# ---------- 7. projection ----------

def test_projection_shape_and_definition(wide_data):
    _, X_c, U, S, Vt = pca_parts(wide_data)
    U_k = get_principal_components(U, 3)
    Z = project_data(X_c, U_k)
    assert Z.shape == (3, 12)
    assert np.isclose(Z[1, 5], U_k[:, 1] @ X_c[:, 5])          # z_ji = u_j^T x_c,i
    assert np.allclose(Z, np.diag(S[:3]) @ Vt[:3], atol=ATOL)   # U_k^T U Sigma V^T = Sigma_k V_k^T


# ---------- 8. reconstruction ----------

def test_full_reconstruction_recovers_X(wide_data):
    mu, X_c, U, _, _ = pca_parts(wide_data)
    U_full = get_principal_components(U, U.shape[1])
    X_hat = reconstruct(project_data(X_c, U_full), mu, U_full)
    assert np.allclose(X_hat, wide_data, atol=ATOL)
    assert np.allclose(X_hat - mu, X_c, atol=ATOL)  # centered data recovered as well


def test_reconstruction_with_rank_components_is_exact(wide_data):
    # rank(X_c) = N - 1 = 11, so 11 components already reconstruct exactly
    mu, X_c, U, _, _ = pca_parts(wide_data)
    U_k = get_principal_components(U, 11)
    assert np.allclose(reconstruct(project_data(X_c, U_k), mu, U_k), wide_data, atol=1e-9)


def test_reconstruction_error_matches_discarded_variance(wide_data):
    """Eckart-Young: ||X_c - U_k U_k^T X_c||_F^2 = sum_{i>k} sigma_i^2."""
    mu, X_c, U, S, _ = pca_parts(wide_data)
    _, cumulative = explained_variance_ratio(S)
    for k in (1, 3, 6):
        U_k = get_principal_components(U, k)
        X_hat = reconstruct(project_data(X_c, U_k), mu, U_k)
        assert np.isclose(np.linalg.norm(wide_data - X_hat) ** 2, np.sum(S[k:] ** 2))
        rel = relative_reconstruction_error(wide_data, X_hat, mu)
        assert np.isclose(rel**2, 1.0 - cumulative[k - 1])


def test_reconstruction_error_decreases_with_k(wide_data):
    mu, X_c, U, _, _ = pca_parts(wide_data)
    errors = []
    for k in range(1, 12):
        U_k = get_principal_components(U, k)
        errors.append(relative_reconstruction_error(wide_data, reconstruct(project_data(X_c, U_k), mu, U_k), mu))
    assert np.all(np.diff(errors) <= ATOL)


def test_reconstruction_error_hand_example():
    original = np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 3.0]])
    reconstructed = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    assert np.allclose(reconstruction_error(original, reconstructed), [3.0, 3.0])  # sqrt(1+4+4), sqrt(9)
    assert np.isclose(reconstruction_error(np.array([3.0, 4.0]), np.zeros(2)), 5.0)


# ---------- real face data (skipped if not downloaded) ----------

@pytest.mark.skipif(not DATASET_DIR.is_dir(), reason="AT&T dataset not downloaded")
def test_real_training_data_pca():
    X, labels = load_dataset(DATASET_DIR)
    X_train, _, _, _ = train_test_split(X, labels, n_test_per_person=3, seed=42)
    mu, X_c, U, S, Vt = pca_parts(X_train)
    assert X_train.shape == (4992, 280)
    assert mu.shape == (4992, 1)
    assert U.shape == (4992, 280) and S.shape == (280,) and Vt.shape == (280, 280)
    assert np.allclose(U.T @ U, np.eye(280), atol=1e-8)
    assert np.allclose(U @ np.diag(S) @ Vt, X_c, atol=1e-8)
    assert S[-1] < 1e-10 * S[0]  # rank(X_c) = 279
    # C u_1 = lambda_1 u_1, evaluated as X_c (X_c^T u_1) / (N - 1) without forming C
    lam = eigenvalues_from_singular_values(S, 280)
    assert np.allclose(X_c @ (X_c.T @ U[:, 0]) / 279, lam[0] * U[:, 0], atol=1e-8)
