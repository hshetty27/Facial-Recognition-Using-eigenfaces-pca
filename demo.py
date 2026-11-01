"""
EigenFaces custom demo: train on photos of a few people, then recognise an unseen photo.

    python demo.py                         interactive menu
    python demo.py train  [options]        PCA on demo_data/<person>/*.jpg, evaluate held-out images
    python demo.py train --all-data        final live-demo model: PCA on ALL images, nothing held out
    python demo.py recognize IMAGE         recognise one image with the trained model
    python demo.py eigenfaces              show mean face and eigenfaces of the trained model
    python demo.py variance                show explained variance of the trained model
    python demo.py sample                  copy 4 AT&T people into data/demo_sample (rehearsal data)

Mathematics (same functions as the main experiment):
    x -> x - mu -> z = U_k^T (x - mu) -> d_i = ||z - z_i||_2 -> identity of argmin_i d_i
See docs/custom_demo.md.
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "demo_data"
DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "demo" / "eigenfaces_model.npz"
SAMPLE_DIR = PROJECT_ROOT / "data" / "demo_sample"
SAMPLE_PEOPLE = {"s1": "Person_A", "s2": "Person_B", "s3": "Person_C", "s4": "Person_D"}
LINE = "=" * 60


def rel(path) -> str:
    """Path relative to the project root when possible (shorter output)."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def show_or_close(show: bool) -> None:
    import matplotlib.pyplot as plt
    if show:
        plt.show()
    plt.close("all")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_train(data_dir: Path, test_dir: Path | None, k: int, n_test: int, seed: int,
              image_shape: tuple[int, int], model_path: Path, show: bool) -> None:
    import numpy as np
    from src.custom_demo import evaluate_split, file_fingerprint, prepare_custom_split, save_model, train_model
    from src.visualization import plot_cumulative_variance, plot_eigenfaces, plot_explained_variance, plot_mean_face
    from src.pca import explained_variance_ratio, get_principal_components

    print(f"\n{LINE}\n TRAINING  (data: {rel(data_dir)})\n{LINE}")
    split = prepare_custom_split(data_dir, image_shape, n_test, seed, test_dir)
    for warning in split["warnings"]:
        print(f"  note, aspect ratio: {warning}")
    people = sorted(set(split["labels_train"].tolist()), key=str.lower)
    d = image_shape[0] * image_shape[1]
    print(f"identities ({len(people)}): {', '.join(people)}")
    for person in people:
        n_tr = int(np.sum(split["labels_train"] == person))
        n_te = int(np.sum(split["labels_test"] == person))
        print(f"  {person:<20} {n_tr:3d} training  {n_te:3d} test")
    print(f"image size {image_shape[0]} x {image_shape[1]}  ->  d = {d}")
    print(f"X_train: {split['X_train'].shape}   X_test: {split['X_test'].shape}")
    if test_dir is None:
        print(f"split: {n_test} random test image(s) per person, seed {seed}; test files never used for PCA")
    else:
        print(f"test images from {rel(test_dir)}; all of {rel(data_dir)} used for training")

    model = train_model(split["X_train"], split["labels_train"], split["train_paths"], k, image_shape)
    print(f"\nPCA via SVD of X_c = X_train - mu (training images only):")
    print(f"  mu {model['mean_face'].shape}, U_k {model['U_k'].shape}, Z_train {model['Z_train'].shape}")
    print(f"  rank(X_c) = {model['rank']}  (at most N_train - 1 = {split['X_train'].shape[1] - 1})")
    if model["k"] < k:
        print(f"  k = {k} requested, but only {model['rank']} non-zero singular values exist -> using k = {model['k']}")
    print(f"  k = {model['k']} components keep {100 * model['explained_variance']:.1f}% of the training variance")

    model["test_files"] = np.array([str(p) for p in split["test_paths"]])
    model["labels_test"] = split["labels_test"]
    model["train_sha256"] = np.array([file_fingerprint(p) for p in split["train_paths"]])
    model["preprocessing"] = (f"EXIF upright -> grayscale (ITU-R 601 luma) -> centre crop to h/w = "
                              f"{image_shape[0]}/{image_shape[1]} -> bilinear resize {image_shape[0]}x{image_shape[1]} "
                              f"-> /255 -> row-major flatten (d = {d})")
    save_model(model, model_path)
    print(f"  model saved: {rel(model_path)}")

    if len(split["test_paths"]):
        evaluation = evaluate_split(model, split)
        print(f"\nHeld-out test images (never used to compute mu or U):")
        for path, truth, result in zip(split["test_paths"], split["labels_test"], evaluation["results"]):
            mark = "correct" if result["predicted"] == truth else "WRONG"
            print(f"  {rel(path):<45} true {truth:<12} predicted {result['predicted']:<12} "
                  f"d = {result['distance']:7.3f}  {mark}")
        correct = int(np.sum(evaluation["predictions"][evaluation["known"]] == split["labels_test"][evaluation["known"]]))
        total = int(evaluation["known"].sum())
        print(f"accuracy on held-out images: {correct}/{total} = {100 * evaluation['accuracy']:.1f}%")
        if total < 20:
            print(f"  (only {total} test images: each one changes the accuracy by {100 / total:.0f} percentage points)")
    else:
        print(f"\nNo held-out test images: all {split['X_train'].shape[1]} images used for PCA "
              f"(final demo model). Use 'recognize' with new photos.")

    shape = model["image_shape"]
    evr, cumulative = explained_variance_ratio(model["singular_values"])
    plot_mean_face(model["mean_face"], shape, save_path=model_path.parent / "demo_mean_face.png")
    plot_eigenfaces(get_principal_components(model["U_k"], min(10, model["k"])), shape,
                    title=f"First eigenfaces of the custom dataset", save_path=model_path.parent / "demo_eigenfaces.png")
    plot_explained_variance(evr[: model["rank"]], save_path=model_path.parent / "demo_explained_variance.png")
    plot_cumulative_variance(cumulative[: model["rank"]], [0.8, 0.9], save_path=model_path.parent / "demo_cumulative_variance.png")
    print(f"figures saved in {rel(model_path.parent)}")
    show_or_close(show)


def cmd_recognize(image_path: Path, model_path: Path, show: bool) -> None:
    import numpy as np
    from src.custom_demo import load_model, recognize_image
    from src.data_loader import load_image
    from src.preprocessing import preprocess_image
    from src.visualization import plot_demo_recognition

    model = load_model(model_path)
    result = recognize_image(model, image_path)
    print(f"\n{LINE}\n RECOGNITION  ({rel(image_path)})\n{LINE}")
    print(f"x in R^{result['x'].size}  ->  z = U_k^T (x - mu) in R^{result['z'].size}  "
          f"->  {len(result['distances'])} distances ||z - z_i||")
    print(f"\nPredicted identity:     {result['predicted']}")
    print(f"Nearest training image: {rel(result['nearest_file'])}")
    print(f"Distance:               {result['distance']:.3f}")
    print("\nSmallest distance to each identity (Euclidean distance in PCA space, not a probability):")
    for person, distance in result["per_identity"].items():
        print(f"  {person:<20} {distance:8.3f}" + ("   <- nearest" if person == result["predicted"] else ""))
    if result["in_training_set"]:
        print("\nNOTE: this exact file is one of the TRAINING images, so its distance is ~0.\n"
              "      This is not a recognition test - use a photo that was not used for training.")

    nearest = Path(result["nearest_file"])
    nearest_vector = (preprocess_image(load_image(nearest), model["image_shape"]) if nearest.exists()
                      else np.zeros(result["x"].size))
    save_path = model_path.parent / f"demo_recognition_{Path(image_path).stem}.png"
    plot_demo_recognition(result["x"], model["mean_face"], result["z"], nearest_vector, result["per_identity"],
                          result["predicted"], model["image_shape"],
                          title=f"{Path(image_path).name}: predicted {result['predicted']} "
                                f"(k = {model['k']}, distance {result['distance']:.2f})",
                          save_path=save_path)
    print(f"\nfigure saved: {rel(save_path)}")
    show_or_close(show)


def cmd_eigenfaces(model_path: Path, show: bool) -> None:
    from src.custom_demo import load_model
    from src.pca import get_principal_components
    from src.visualization import plot_eigenfaces, plot_mean_face

    model = load_model(model_path)
    plot_mean_face(model["mean_face"], model["image_shape"], save_path=model_path.parent / "demo_mean_face.png")
    plot_eigenfaces(get_principal_components(model["U_k"], min(10, model["k"])), model["image_shape"],
                    title="First eigenfaces of the custom dataset", save_path=model_path.parent / "demo_eigenfaces.png")
    print(f"mean face and eigenfaces saved in {rel(model_path.parent)}")
    show_or_close(show)


def cmd_variance(model_path: Path, show: bool) -> None:
    from src.custom_demo import load_model
    from src.pca import explained_variance_ratio
    from src.visualization import plot_cumulative_variance, plot_explained_variance

    model = load_model(model_path)
    S = model["singular_values"][: model["rank"]]
    evr, cumulative = explained_variance_ratio(S)
    print(f"rank = {model['rank']}, k = {model['k']}: {100 * model['explained_variance']:.1f}% of the variance kept")
    print("  i   EVR_i   cumulative")
    for i in range(min(10, len(evr))):
        print(f"{i + 1:3d}  {100 * evr[i]:5.1f}%   {100 * cumulative[i]:5.1f}%")
    plot_explained_variance(evr, save_path=model_path.parent / "demo_explained_variance.png")
    plot_cumulative_variance(cumulative, [0.8, 0.9], save_path=model_path.parent / "demo_cumulative_variance.png")
    show_or_close(show)


def cmd_sample() -> None:
    """Copy 4 AT&T people into data/demo_sample/ as rehearsal data in the custom-demo layout."""
    from src.data_loader import download_att_faces
    source = download_att_faces()
    for att_name, demo_name in SAMPLE_PEOPLE.items():
        target = SAMPLE_DIR / demo_name
        target.mkdir(parents=True, exist_ok=True)
        for image in sorted((source / att_name).glob("*.pgm")):
            shutil.copy2(image, target / f"{int(image.stem):02d}.pgm")
    print(f"Sample data: {rel(SAMPLE_DIR)}  ({', '.join(f'{v} = AT&T {k}' for k, v in SAMPLE_PEOPLE.items())})")
    print(f"Try:  python demo.py train --data-dir {rel(SAMPLE_DIR)}")


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

def ask(prompt: str, default: str) -> str:
    answer = input(f"{prompt} [{default}]: ").strip()
    return answer or default


def menu(args: argparse.Namespace) -> None:
    from src.custom_demo import load_model
    while True:
        print(f"\n{LINE}\n{'EIGENFACES FACE RECOGNITION':^60}\n{LINE}")
        print(" 1. Train custom dataset\n 2. Recognize a face\n 3. Show mean face and eigenfaces\n"
              " 4. Show explained variance\n 5. Exit")
        try:
            choice = input("\nChoose 1-5: ").strip()
            if choice == "1":
                data_dir = Path(ask("Dataset folder", rel(args.data_dir)))
                k = int(ask("Number of components k", str(args.k)))
                n_test = int(ask("Held-out test images per person", str(args.n_test)))
                cmd_train(PROJECT_ROOT / data_dir if not data_dir.is_absolute() else data_dir, None, k, n_test,
                          args.seed, (args.height, args.width), args.model, show=not args.no_show)
            elif choice == "2":
                model = load_model(args.model)
                test_files = list(model.get("test_files", []))
                if test_files:
                    print("Held-out test images (not used for training):")
                    for i, f in enumerate(test_files, start=1):
                        print(f"  {i:2d}. {rel(f)}")
                answer = input("Enter a number from the list, or a path to an image: ").strip().strip('"')
                path = Path(test_files[int(answer) - 1]) if answer.isdigit() and test_files else Path(answer)
                cmd_recognize(path if path.is_absolute() else PROJECT_ROOT / path, args.model, show=not args.no_show)
            elif choice == "3":
                cmd_eigenfaces(args.model, show=not args.no_show)
            elif choice == "4":
                cmd_variance(args.model, show=not args.no_show)
            elif choice == "5":
                return
            else:
                print("Please enter a number from 1 to 5.")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        except FileNotFoundError as error:
            print(f"Not found: {error}. (Train a model first, and check the path.)")
        except (ValueError, IndexError) as error:
            print(f"Error: {error}")


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EigenFaces custom demo (PCA + nearest neighbour).")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="trained model file (.npz)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="folder with one sub-folder per person")
    parser.add_argument("--test-dir", type=Path, default=None,
                        help="optional folder of unseen test images (same layout); then all of --data-dir is training")
    parser.add_argument("--k", type=int, default=30, help="number of principal components (reduced to rank if larger)")
    parser.add_argument("--n-test", type=int, default=3, help="random held-out test images per person (ignored with --test-dir)")
    parser.add_argument("--all-data", action="store_true",
                        help="train on every image of --data-dir (same as --n-test 0): the final live-demo model")
    parser.add_argument("--seed", type=int, default=42, help="random seed of the train/test split")
    parser.add_argument("--height", type=int, default=78, help="image height after resizing")
    parser.add_argument("--width", type=int, default=64, help="image width after resizing")
    parser.add_argument("--no-show", action="store_true", help="save figures without opening windows")
    sub = parser.add_subparsers(dest="command")
    train = sub.add_parser("train", help="train on --data-dir and evaluate held-out images")
    train.add_argument("--all-data", action="store_true", default=argparse.SUPPRESS,
                       help="train on every image of --data-dir (same as --n-test 0)")
    recognize = sub.add_parser("recognize", help="recognise one image")
    recognize.add_argument("image", type=Path)
    sub.add_parser("eigenfaces", help="show mean face and eigenfaces")
    sub.add_parser("variance", help="show explained variance")
    sub.add_parser("sample", help="create rehearsal data from 4 AT&T people")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    show = not args.no_show
    try:
        if args.command is None:
            menu(args)
        elif args.command == "train":
            if args.all_data and args.test_dir is not None:
                raise ValueError("--all-data and --test-dir cannot be combined")
            cmd_train(args.data_dir, args.test_dir, args.k, 0 if args.all_data else args.n_test, args.seed,
                      (args.height, args.width), args.model, show)
        elif args.command == "recognize":
            cmd_recognize(args.image, args.model, show)
        elif args.command == "eigenfaces":
            cmd_eigenfaces(args.model, show)
        elif args.command == "variance":
            cmd_variance(args.model, show)
        elif args.command == "sample":
            cmd_sample()
    except (FileNotFoundError, ValueError) as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    main()
