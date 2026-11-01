"""
Projection onto the PCA subspace, and reconstruction from it.

Convention: X has shape (d, N); U_k has shape (d, k) with orthonormal columns;
mu has shape (d, 1).

    Z     = U_k^T X_c                 coordinates of each face along u_1, ..., u_k
    X_hat = U_k Z + mu = U_k U_k^T X_c + mu

U_k U_k^T is the orthogonal projection matrix onto span(u_1, ..., u_k).
"""

import numpy as np


def project_data(X_centered: np.ndarray, U_k: np.ndarray) -> np.ndarray:
    """
    Coordinates of centered faces along the principal directions:  Z = U_k^T X_c.

    Input:  X_c of shape (d, N), U_k of shape (d, k).
    Output: Z of shape (k, N); column i holds the k "eigenface weights" of face i.

    Interpretation: since the u_j are orthonormal, z_ji = u_j^T (x_i - mu) is just a
    dot product. This is the dimensionality reduction: d numbers -> k numbers.
    """
    if X_centered.shape[0] != U_k.shape[0]:
        raise ValueError(f"X_c has {X_centered.shape[0]} rows but U_k has {U_k.shape[0]}")
    return U_k.T @ X_centered


def project_new_face(face_vector: np.ndarray, mean_face: np.ndarray, principal_components: np.ndarray) -> np.ndarray:
    """
    Project a single (unseen) face:  z = U_k^T (x - mu).

    Convention: one face is a 1-D vector.
    Input:  face_vector x of shape (d,);
            mean_face mu of shape (d, 1)  - the TRAINING mean;
            principal_components U_k of shape (d, k) - from the TRAINING SVD.
    Output: z of shape (k,): the k PCA coordinates of the face.
    Inputs of any other shape (e.g. x as (d, 1) or (1, d)) are rejected, not reshaped.

    Interpretation: the new face is centered with the training mean and expressed in
    the training eigenface basis, so z lives in the same k-dim space as the columns
    of Z_train = U_k^T X_c. Equivalent to one column of project_data.
    """
    d, k = principal_components.shape
    if face_vector.shape != (d,):
        raise ValueError(f"face_vector must have shape ({d},), got {face_vector.shape}")
    if mean_face.shape != (d, 1):
        raise ValueError(f"mean_face must have shape ({d}, 1), got {mean_face.shape}")
    x_c = face_vector - mean_face[:, 0]  # x - mu, shape (d,)
    return principal_components.T @ x_c  # U_k^T (x - mu), shape (k,)


def reconstruct(Z: np.ndarray, mu: np.ndarray, U_k: np.ndarray) -> np.ndarray:
    """
    Map PCA coordinates back to pixel space:  X_hat = U_k Z + mu.

    Input:  Z of shape (k, N), mu of shape (d, 1), U_k of shape (d, k).
    Output: X_hat of shape (d, N).

    With Z = U_k^T X_c this is X_hat = U_k U_k^T (X - mu) + mu, the orthogonal
    projection of each face onto the affine subspace mu + span(u_1, ..., u_k).
    With k = rank(X_c), X_hat = X exactly (up to rounding).
    """
    if mu.shape != (U_k.shape[0], 1):
        raise ValueError(f"mu must have shape ({U_k.shape[0]}, 1), got {mu.shape}")
    return U_k @ Z + mu


def reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """
    Euclidean reconstruction error of each face:  e_i = || x_i - x_hat_i ||_2.

    Input:  original and reconstructed, both of shape (d, N), or both of shape (d,).
    Output: errors of shape (N,), or a scalar for a single vector.

    Interpretation: x_i - x_hat_i is the part of the centered face that is orthogonal
    to span(U_k); e_i is its length, i.e. what is lost by keeping only k components.
    Units: pixel intensities on the [0, 1] scale.
    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")
    return np.linalg.norm(original - reconstructed, axis=0)


def relative_reconstruction_error(original: np.ndarray, reconstructed: np.ndarray, mu: np.ndarray) -> float:
    """
    Normalized Frobenius error of a whole data matrix:

        || X - X_hat ||_F / || X - mu ||_F  =  || X_c - U_k U_k^T X_c ||_F / || X_c ||_F

    Input:  original X and reconstructed X_hat of shape (d, N); mu of shape (d, 1).
    Output: a number in [0, 1]. 0 = perfect, 1 = no better than the mean face alone.

    The denominator is the size of the CENTERED data, so this measures how much of the
    variation around the mean face is lost. For the training data it satisfies
        error^2 = sum_{i>k} sigma_i^2 / sum_j sigma_j^2 = 1 - cumulative_EVR[k - 1].
    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")
    return float(np.linalg.norm(original - reconstructed) / np.linalg.norm(original - mu))
