"""
Face recognition by nearest neighbour in PCA space.

Every face is a point in R^k (its PCA coordinates). A test face is assigned the
identity of the training face whose point is closest in Euclidean distance:

    z_test = U_k^T (x_test - mu)            (training mu and U_k)
    d_i    = || z_test - z_i ||_2           for every training column z_i of Z_train
    label  = labels_train[ argmin_i d_i ]

No model is trained beyond PCA itself: the "database" is simply Z_train.
"""

import numpy as np

from src.projection import project_new_face


def euclidean_distances(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """
    Distances from one point to every column of a matrix:
        d_i = || z - z_i ||_2 = sqrt( sum_j (z_j - z_ji)^2 ).

    Input:  query z of shape (k,); database Z of shape (k, N) (one point per column).
    Output: distances of shape (N,).
    """
    if query.ndim != 1 or database.ndim != 2 or database.shape[0] != query.shape[0]:
        raise ValueError(f"Need query (k,) and database (k, N); got {query.shape} and {database.shape}")
    differences = database - query[:, np.newaxis]  # column i is z_i - z, shape (k, N)
    return np.sqrt(np.sum(differences**2, axis=0))


def recognize(
    query_projection: np.ndarray,
    training_projections: np.ndarray,
    labels: np.ndarray,
) -> tuple[str, int, float]:
    """
    Nearest-neighbour identification:  i* = argmin_i || z - z_i ||_2,  prediction = labels[i*].

    Input:  query_projection z of shape (k,); training_projections Z_train of shape (k, N);
            labels of shape (N,), labels[i] = person of column i.
    Output: (predicted label, index i* of the nearest training face, distance d_{i*}).
    If two training faces are exactly equally close, the one with the smaller index wins.
    """
    if training_projections.shape[1] != len(labels):
        raise ValueError("Need one label per column of training_projections")
    distances = euclidean_distances(query_projection, training_projections)
    nearest = int(np.argmin(distances))
    return str(labels[nearest]), nearest, float(distances[nearest])


def recognize_faces(
    X_test: np.ndarray,
    mean_face: np.ndarray,
    principal_components: np.ndarray,
    training_projections: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recognise every column of X_test, one face at a time:
        z_j = U_k^T (x_j - mu),  then recognize(z_j, Z_train, labels).

    Input:  X_test (d, M); TRAINING mean_face (d, 1) and principal_components U_k (d, k);
            training_projections Z_train (k, N); labels (N,).
    Output: predictions (M,), nearest training indices (M,), nearest distances (M,).

    Each test face is processed on its own with training quantities only, so the
    result for one test face can never depend on any other test face.
    """
    predictions, indices, distances = [], [], []
    for j in range(X_test.shape[1]):
        z = project_new_face(X_test[:, j], mean_face, principal_components)
        label, index, distance = recognize(z, training_projections, labels)
        predictions.append(label)
        indices.append(index)
        distances.append(distance)
    return np.array(predictions), np.array(indices), np.array(distances)


def accuracy(predictions: np.ndarray, true_labels: np.ndarray) -> float:
    """
    Fraction of correct predictions:  (number of j with prediction_j = true_j) / M.

    Input:  two label arrays of equal length M >= 1.
    Output: float in [0, 1].
    """
    predictions, true_labels = np.asarray(predictions), np.asarray(true_labels)
    if predictions.shape != true_labels.shape or predictions.size == 0:
        raise ValueError("Need two non-empty label arrays of equal length")
    return float(np.mean(predictions == true_labels))


def confusion_matrix(true_labels: np.ndarray, predictions: np.ndarray, classes: list[str]) -> np.ndarray:
    """
    Count matrix M with M[a, b] = number of test faces of person classes[a] predicted as classes[b].

    Rows = true identity, columns = predicted identity; the diagonal holds correct
    predictions, so trace(M) / sum(M) = accuracy.
    """
    position = {c: i for i, c in enumerate(classes)}
    counts = np.zeros((len(classes), len(classes)), dtype=int)
    for true, predicted in zip(true_labels, predictions):
        counts[position[true], position[predicted]] += 1
    return counts
