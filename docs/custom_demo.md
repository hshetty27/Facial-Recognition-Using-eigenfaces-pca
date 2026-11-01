# Custom Demo: Recognising Your Own Faces

The custom demo trains the **same** eigenfaces pipeline as the AT&T experiment on photos you provide (for example, of 4 consenting friends), then recognises a photo it has not seen. It uses the same preprocessing, the same PCA via SVD, the same projection and the same nearest-neighbour rule. Nothing new is learned.

> **Consent and privacy.** Only use photos of people who have agreed. `demo_data/`, `demo_test/`, `data/demo_sample/` and `outputs/demo/` (the trained model and figures) are excluded by `.gitignore`. Do not commit or share them.

---

## 1. Folder structure

The folder name is the identity label. Any number of people (two or more) is allowed.

```
eigenfaces-pca/
└── demo_data/
    ├── Friend_A/
    │   ├── 01.jpg
    │   ├── 02.jpg
    │   └── ...            (10 or more photos recommended)
    ├── Friend_B/
    ├── Friend_C/
    └── Friend_D/
```

Optionally, keep separate unseen test photos in the same layout:

```
eigenfaces-pca/
└── demo_test/
    ├── Friend_A/
    │   └── t1.jpg
    └── ...
```

## 2. Adding photos

The eigenfaces method compares faces **pixel by pixel**, so the photos must be consistent. For good results:

| Do | Why |
|---|---|
| Crop each photo to the head: face centred, eyes at about the same height in every photo | There is no automatic face detection. A shifted face is a different vector. |
| Use a portrait crop with height : width ≈ 1.22 (e.g. 640 × 780 px) | Images are centre-cropped to 78 : 64 and then resized to 78 × 64 (never stretched). Other shapes lose part of their border, and the demo prints a note for any image off by more than 10 %. Keep the face centred and at the same size in every photo, including test photos. |
| Frontal pose, similar lighting, plain background, for everyone | Lighting and background differences become the largest "variation", and PCA will model them instead of identity. |
| Vary the expression a little (smile, neutral, glasses on and off) | This gives PCA real within-person variation, as in AT&T. |
| JPG, PNG, BMP, TIFF, WebP or PGM | Read by Pillow. **HEIC (iPhone) is not supported**, so export as JPG. |

Phone photos taken in portrait mode are rotated upright automatically (EXIF orientation).

## 3. How training works

```bash
python demo.py train                          # uses demo_data/, k = 30, 3 test photos per person, seed 42
```

1. All photos are loaded and preprocessed exactly like the AT&T images: EXIF orientation → grayscale → centre crop to 78 : 64 → resize to 78 × 64 → divide by 255 → flatten to x ∈ ℝ⁴⁹⁹².
2. Train/test split. For each person, `--n-test` photos (default 3) are held out at random with a fixed `--seed`. This uses the same `split_indices` as the main experiment. The demo refuses to run if a test photo is byte-identical to a training photo.
3. PCA on the **training photos only**:
   μ = (1/N) X 1 → X_c = X − μ → X_c = UΣVᵀ → U_k → Z_train = U_kᵀ X_c.
4. **k is limited by the rank.** With N training photos, rank(X_c) ≤ N − 1. For example, 4 people × 7 photos gives 28 training images, so rank 27. If you ask for k = 30, the demo uses k = 27 and says so.
5. Every held-out photo is recognised, and the accuracy is printed.
6. The model (μ, U_k, Z_train, labels, file names, singular values, image size, k) is saved locally to `outputs/demo/eigenfaces_model.npz`. The mean face, eigenfaces and variance figures are saved next to it.

**Never evaluate on a training photo.** Its distance to itself is 0, which proves nothing. The demo enforces this for the held-out set, and `recognize` prints a warning if you give it a training photo.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--data-dir` | `demo_data` | one sub-folder per person |
| `--k` | 30 | number of principal components (reduced to the rank if larger) |
| `--n-test` | 3 | random held-out photos per person (0 = train on everything) |
| `--all-data` | off | final live-demo model: PCA on **all** photos of `--data-dir` (same as `--n-test 0`); may also be written after `train` |
| `--seed` | 42 | seed of the random split |
| `--test-dir` | none | use this folder as the unseen test set; then **all** of `--data-dir` is training |
| `--height`, `--width` | 78, 64 | image size after resizing (keep the 1.22 ratio) |
| `--model` | `outputs/demo/eigenfaces_model.npz` | where the trained model is stored |
| `--no-show` | off | save figures without opening windows |

Options go **before** the command, e.g. `python demo.py --k 10 --n-test 2 train`.

k = 30 is only a default. In the AT&T experiment, 30 tied with 50 for the best accuracy, but more retained variance did not mean better recognition (k = 66 was slightly worse). Try several k values live, for example `--k 5`, `--k 10` and `--k 20`.

## 4. Unseen test photos

Choose one of three ways:
- **Automatic** (simplest): keep about 10 photos per person in `demo_data/`; `--n-test 3` holds 3 out.
- **Separate folder:** put extra photos in `demo_test/<person>/`, then run
  `python demo.py --test-dir demo_test train`.
- **Live:** train with `--n-test 0`, take a new photo during the presentation, crop it, then run
  `python demo.py recognize new_photo.jpg`.

## 5. Commands

```bash
python demo.py                                   # interactive menu (recommended for the live demo)
python demo.py train                             # train on demo_data/ and evaluate held-out photos
python demo.py --k 10 train                      # same, with k = 10
python demo.py --test-dir demo_test train        # all of demo_data/ for training, demo_test/ for testing
python demo.py --k 10 train --all-data          # final live-demo model: all photos, nothing held out
python demo.py recognize demo_test/Friend_C/t1.jpg
python demo.py eigenfaces                        # mean face + eigenfaces of the trained model
python demo.py variance                          # explained variance of the trained model
python demo.py sample                            # rehearsal data: 4 AT&T people in data/demo_sample/
python demo.py --data-dir data/demo_sample train
```

The interactive menu:

```
============================================================
                EIGENFACES FACE RECOGNITION
============================================================
 1. Train custom dataset
 2. Recognize a face           <- lists the held-out photos; pick a number or type a path
 3. Show mean face and eigenfaces
 4. Show explained variance
 5. Exit
```

**Rehearsal without friends' photos:** `python demo.py sample` copies 4 AT&T people (s1–s4) into `data/demo_sample/Person_A … Person_D`. This is a reliable fallback for the live presentation.

## 6. Output

`train` prints the identities and counts, the matrix shapes, rank(X_c), the k used and its retained variance, then one line per held-out photo and the accuracy. Rehearsal-data example:

```
X_train: (4992, 28)   X_test: (4992, 12)
  mu (4992, 1), U_k (4992, 27), Z_train (27, 28)
  rank(X_c) = 27  (at most N_train - 1 = 27)
  k = 30 requested, but only 27 non-zero singular values exist -> using k = 27
accuracy on held-out images: 12/12 = 100.0%
  (only 12 test images: each one changes the accuracy by 8 percentage points)
```

`recognize` prints:

```
Predicted identity:     Person_C
Nearest training image: data\demo_sample\Person_C\04.pgm
Distance:               3.741

Smallest distance to each identity (Euclidean distance in PCA space, not a probability):
  Person_A               12.898
  Person_B               11.161
  Person_C                3.741   <- nearest
  Person_D                7.802
```

It also saves `outputs/demo/demo_recognition_<name>.png`, which shows: input face → mean face → centered face → the k coordinates of z → nearest training face, plus the distance to each person.

**Distances are not probabilities.** No confidence percentage is shown, because the method does not produce one. A small distance means "close in PCA space", and values are only comparable for the same model and k.

## 7. The mathematics underneath

```
image  --grayscale, centre crop 78:64, resize 78x64, /255, flatten-->  x in R^4992                 (preprocess_image)
x      --minus training mean face-->              x - mu                         (project_new_face)
       --project onto training eigenfaces-->      z = U_k^T (x - mu) in R^k      (project_new_face)
z      --compare with every training face-->      d_i = ||z - z_i||_2            (euclidean_distances)
       --nearest-->                               identity = label(argmin_i d_i) (recognize)
```

Training uses `compute_mean_face`, `center_data`, `compute_svd`, `get_principal_components` and `project_data` from `src/pca.py` and `src/projection.py`. These are the same functions, and the same equations, as the main experiment (see `docs/equation_to_code.md`).

## 8. What to expect, honestly

- With a handful of people and consistently cropped photos, accuracy is usually high. With 12 test photos, though, each mistake costs 8 percentage points.
- Inconsistent framing, lighting or background is the most common cause of errors. Eigenfaces model pixel variation, and they cannot tell lighting changes from identity.
- Anyone who is not in `demo_data/` is still assigned the *nearest known* person. This is closed-set recognition, with no "unknown" answer.


## Diagnostics (`experiments/custom_diagnostics.py`)

```bash
python -m experiments.custom_diagnostics          # demo_data/ + unlabelled images in mystery/
```

Measures the official method without changing it. One SVD per training set; every k uses `U[:, :k]`.

| Output (in `outputs/demo/`) | Content |
|---|---|
| `k_comparison.csv`, `accuracy_vs_k_custom.png` | variance retained and accuracy for k = 2 … 30: held-out 12 (48/12 split, seed 42), and leave-one-out over all 60 |
| `mystery_predictions.csv` | for each k and each mystery image: nearest training face (official) and nearest class mean m_c = U_kᵀ(μ_c − μ) (diagnostic only), with distances |
| `custom_pca_space.png` | training faces in the (PC1, PC2) plane |
| `framing_stats.csv`, `framing_comparison.png`, `all_preprocessed_78x64.png` | the final 78 × 64 vectors, and how far each image lies from the face space: ‖(x − μ) − U_kU_kᵀ(x − μ)‖ / ‖x − μ‖ |

A test photo framed differently from the training photos (face smaller, more background) lies far from the face space (residual ≈ 0.8 instead of ≈ 0.25). Its nearest neighbour is then decided by framing, not identity. Crop test photos exactly like the training photos.
