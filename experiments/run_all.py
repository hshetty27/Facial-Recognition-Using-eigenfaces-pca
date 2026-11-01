"""
Run the complete AT&T experiment and regenerate every result file and figure.

    python -m experiments.run_all

Steps (each is also runnable on its own):
    1. experiments.inspect_dataset        dataset, X (4992 x 400), split, data figures
    2. experiments.pca_toy_example        PCA by hand on a 2 x 5 matrix (printed only)
    3. experiments.run_pca                mean face, SVD, eigenvalues, eigenfaces, reconstructions
    4. experiments.run_recognition        recognition for k = 5, 10, 20, 30, 50, 66
    5. experiments.recognize_single_face  one test face step by step (test face 0, k = 66)

Everything is deterministic (fixed split seed 42; no other randomness), so repeated runs
give identical numbers.
"""

import matplotlib

matplotlib.use("Agg")

from experiments import inspect_dataset, pca_toy_example, recognize_single_face, run_pca, run_recognition


def banner(title: str) -> None:
    print(f"\n{'#' * 70}\n# {title}\n{'#' * 70}")


def main() -> None:
    banner("1/5  Dataset and data matrix")
    inspect_dataset.main(show=False)
    banner("2/5  PCA toy example")
    pca_toy_example.main()
    banner("3/5  PCA of the training faces")
    run_pca.main()
    banner("4/5  Recognition for several k")
    run_recognition.main()
    banner("5/5  Single-face demonstration")
    recognize_single_face.main(test_index=0, k=66)
    print("\nAll results in outputs/results/, all figures in outputs/figures/.")


if __name__ == "__main__":
    main()
