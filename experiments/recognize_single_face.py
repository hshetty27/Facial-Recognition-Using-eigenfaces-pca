"""
Phase 4 demonstration: ONE test face through the whole pipeline.

    image -> vector x -> x - mu -> z = U_k^T (x - mu) -> distances ||z - z_i|| -> argmin -> identity

Run from the project root:
    python -m experiments.recognize_single_face                 # test face 0, k = 66
    python -m experiments.recognize_single_face --index 7 --k 20

The default k = 66 is the smallest k retaining 90 % of the training variance
(Phase 3), chosen before any recognition result was seen.
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import numpy as np

from src.data_loader import DATASET_DIR, PROJECT_ROOT, download_att_faces, load_dataset, train_test_split
from src.pca import center_data, compute_mean_face, compute_svd, get_principal_components
from src.preprocessing import IMAGE_SHAPE
from src.projection import project_data, project_new_face
from src.recognition import euclidean_distances, recognize
from src.visualization import plot_single_recognition

N_TEST_PER_PERSON = 3
SEED = 42
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def main(test_index: int, k: int) -> None:
    np.set_printoptions(precision=3, suppress=True)
    download_att_faces()
    X, labels = load_dataset(DATASET_DIR, IMAGE_SHAPE)
    X_train, labels_train, X_test, labels_test = train_test_split(X, labels, N_TEST_PER_PERSON, SEED)

    # PCA from the TRAINING faces only
    mu = compute_mean_face(X_train)
    X_c = center_data(X_train, mu)
    U, _, _ = compute_svd(X_c)
    U_k = get_principal_components(U, k)
    Z_train = project_data(X_c, U_k)
    print(f"Training: X_train {X_train.shape} -> mu {mu.shape}, U_k {U_k.shape}, Z_train {Z_train.shape}\n")

    # One unseen face
    x = X_test[:, test_index]
    true_label = labels_test[test_index]
    print(f"1. test image -> vector x:          shape {x.shape}")
    x_c = x - mu[:, 0]
    print(f"2. subtract training mean x - mu:   shape {x_c.shape}")
    z = project_new_face(x, mu, U_k)
    print(f"3. project z = U_k^T (x - mu):      shape {z.shape}")
    print(f"   first 5 coordinates: {z[:5]}")
    distances = euclidean_distances(z, Z_train)
    print(f"4. distances ||z - z_i||:           shape {distances.shape} (one per training face)")
    predicted_label, nearest, nearest_distance = recognize(z, Z_train, labels_train)
    print(f"5. argmin_i ||z - z_i||:            training column {nearest}")
    print(f"6. identity of that training face:  {predicted_label}\n")

    print(f"True identity:      {true_label}")
    print(f"Predicted identity: {predicted_label}   ({'correct' if predicted_label == true_label else 'WRONG'})")
    print(f"Nearest distance:   {nearest_distance:.4f}")
    own = distances[labels_train == true_label]
    others = distances[labels_train != true_label]
    print(f"Distances to the 7 training faces of {true_label}: min {own.min():.3f}, max {own.max():.3f}")
    print(f"Distances to the 273 training faces of others: min {others.min():.3f}")

    save_path = FIGURES_DIR / f"recognition_single_face_test{test_index}_k{k}.png"
    plot_single_recognition(x, X_train[:, nearest], distances, labels_train, true_label, predicted_label,
                            IMAGE_SHAPE, k, save_path=save_path)
    print(f"\nFigure saved to {save_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognise one test face step by step.")
    parser.add_argument("--index", type=int, default=0, help="column of X_test (0-119)")
    parser.add_argument("--k", type=int, default=66, help="number of principal components")
    args = parser.parse_args()
    main(args.index, args.k)
