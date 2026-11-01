"""
Phase 2 experiment: load the AT&T faces, build X, verify it, split it, and save figures.

Run from the project root:

    python -m experiments.inspect_dataset          # save figures only
    python -m experiments.inspect_dataset --show   # also open the figure windows

No PCA is performed here.
"""

import argparse

import matplotlib
import numpy as np

from src.data_loader import (
    DATASET_DIR, PROJECT_ROOT, discover_images, download_att_faces,
    load_dataset, load_image, train_test_split,
)
from src.preprocessing import (
    IMAGE_SHAPE, format_report, image_to_array, vector_to_image, verify_data_matrix,
)

N_TEST_PER_PERSON = 3   # 10 images per person -> 7 train / 3 test
SEED = 42

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"


def main(show: bool) -> None:
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.visualization import plot_faces, plot_one_per_person, plot_vector_roundtrip

    download_att_faces()

    # 1. Original images as stored on disk
    paths, _ = discover_images(DATASET_DIR)
    original = load_image(paths[0])
    print(f"Original image: {original.size[0]} x {original.size[1]} (w x h), mode '{original.mode}'")

    # 2. Data matrix and labels
    X, labels = load_dataset(DATASET_DIR, IMAGE_SHAPE)
    summary = verify_data_matrix(X, labels, IMAGE_SHAPE)
    report = format_report(summary)
    print("\n" + report)
    assert X.shape[1] == len(labels)
    assert X.shape[0] == IMAGE_SHAPE[0] * IMAGE_SHAPE[1] == 4992

    # 3. Train/test split
    X_train, y_train, X_test, y_test = train_test_split(X, labels, N_TEST_PER_PERSON, SEED)
    split_report = "\n".join([
        f"Split (seed={SEED}, {N_TEST_PER_PERSON} test images per person):",
        f"  X_train shape:      {X_train.shape}   labels_train shape: {y_train.shape}",
        f"  X_test shape:       {X_test.shape}   labels_test shape:  {y_test.shape}",
        f"  people in train:    {len(np.unique(y_train))}",
        f"  people in test:     {len(np.unique(y_test))}",
    ])
    print("\n" + split_report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "dataset_report.txt").write_text(
        f"Dataset: AT&T Database of Faces ({DATASET_DIR.relative_to(PROJECT_ROOT)})\n"
        f"Original image size: {original.size[0]} x {original.size[1]} (w x h)\n\n"
        + report + "\n\n" + split_report + "\n",
        encoding="utf-8",
    )

    # 4. Figures
    sample_paths = paths[::10][:16]  # image 1 of the first 16 people
    plot_faces(
        [image_to_array(load_image(p)) for p in sample_paths],
        [f"{p.parent.name}/{p.name}" for p in sample_paths],
        suptitle="Sample original images (92 x 112)",
        save_path=FIGURES_DIR / "sample_original_faces.png",
    )
    plot_faces(
        [vector_to_image(X[:, j], IMAGE_SHAPE) for j in range(10)],
        [f"column {j}" for j in range(10)],
        n_cols=10,
        suptitle=f"All 10 images of person {labels[0]}: columns 0-9 of X reshaped to 78 x 64",
        save_path=FIGURES_DIR / "one_person_all_images.png",
    )
    plot_vector_roundtrip(
        image_to_array(original), X[:, 0], IMAGE_SHAPE,
        title=f"{paths[0].parent.name}/{paths[0].name}: image -> vector -> image",
        save_path=FIGURES_DIR / "vector_roundtrip.png",
    )
    plot_one_per_person(X, labels, IMAGE_SHAPE, save_path=FIGURES_DIR / "one_face_per_person.png")
    print(f"\nFigures saved to {FIGURES_DIR.relative_to(PROJECT_ROOT)}")

    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--show", action="store_true", help="open figure windows")
    main(parser.parse_args().show)
