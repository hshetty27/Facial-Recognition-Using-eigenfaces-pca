# Final Project Audit

**Project:** EigenFaces: Facial Recognition Through PCA, Eigenvectors and Projection
**Audit date:** 2026-10-05. All values below were regenerated from the code on this date and match the committed result files.

---

## 1. Project objective

The project demonstrates the linear algebra of PCA (vectors, matrices, mean-centering, covariance, eigenvalues, eigenvectors, orthogonality, SVD, projection, dimensionality reduction, Euclidean distance) through the classical *eigenfaces* face-recognition method. Face recognition is the application; the mathematics is the subject. It is an educational demonstration, not a production recognition system.

## 2. Mathematical scope

Implemented with NumPy only. `numpy.linalg.svd` is the only library linear-algebra routine in the core.

Not used anywhere in the code:
- neural networks, CNNs or deep learning;
- pretrained models or embeddings;
- scikit-learn;
- OpenCV, dlib, MediaPipe or `face_recognition`;
- cloud APIs.

An import audit (`grep` over `src/`, `experiments/`, `tests/`, `demo.py`) finds only `numpy`, `matplotlib`, `PIL`, `pytest` and the standard library. A fresh virtual environment containing only `requirements.txt` runs everything (§13).

## 3. Dataset

AT&T Database of Faces (formerly ORL), Olivetti Research Laboratory, 1992–94.
- Source: `https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip` (3.8 MB, no login).
- The zip's SHA-256 is verified before extraction (`src/data_loader.py::download_att_faces`).

| Property | Value (verified from code) |
|---|---|
| people | 40 (`s1` … `s40`) |
| images | 400 (10 per person) |
| original size | 92 × 112 (w × h), 8-bit grayscale PGM |
| pixel range after preprocessing | [0.0196, 0.9098] |

## 4. Preprocessing

`src/preprocessing.py::preprocess_image`. Every image, including custom-demo photos, goes through the same steps:
1. Load with Pillow, applying the EXIF orientation if present. This is a no-op for AT&T.
2. Convert to grayscale (Pillow mode "L"; luma 0.299R + 0.587G + 0.114B).
3. Centre-crop to height : width = 78 : 64, then resize bilinearly to **78 × 64** (h × w). For AT&T the crop is a no-op and the scale factors are 78/112 = 64/92 = 0.696. For other shapes (e.g. 448 × 672 phone photos) the centre crop makes both scale factors equal, so faces are not stretched.
4. Divide by 255, giving values in [0, 1].
5. Flatten row-major into x ∈ ℝ⁴⁹⁹².

No histogram equalisation, alignment, augmentation or feature extraction.

## 5. Matrix shapes

| Object | Shape |
|---|---|
| X (all images) | (4992, 400) |
| labels | (400,) |
| X_train / labels_train | (4992, 280) / (280,) |
| X_test / labels_test | (4992, 120) / (120,) |
| μ | (4992, 1) |
| X_c | (4992, 280) |
| U, S, Vᵀ | (4992, 280), (280,), (280, 280) |
| U_k | (4992, k) |
| Z_train | (k, 280) |
| z (one face) | (k,) |
| distances | (280,) |

**Split:** for each person, 3 test and 7 training images, chosen with `numpy.random.default_rng(42)` (`split_indices`). Every person appears in both sets, and the two sets are disjoint.

## 6. PCA / SVD mathematics (verified on the real training data)

| Statement | Verified value |
|---|---|
| μ = (1/N) X **1** | `compute_mean_face` |
| max \|row mean of X_c\| | 6.6 × 10⁻¹⁶ |
| ‖X_c **1**‖ (columns sum to zero) | 2.3 × 10⁻¹² (‖X_c‖_F ≈ 177) |
| ‖X_c − UΣVᵀ‖_F / ‖X_c‖_F | 2.4 × 10⁻¹⁵ |
| max \|UᵀU − I\| | 2.8 × 10⁻¹⁵ |
| ‖C U₁₀ − U₁₀Λ₁₀‖ / ‖U₁₀Λ₁₀‖, with λ_i = σ_i²/(N−1) | 1.6 × 10⁻¹⁵ (C applied as X_c(X_cᵀu)/(N−1), never formed) |
| \|Σλ_i − Σ pixel variances\| (trace(C)) | 1.4 × 10⁻¹⁴ |
| \|ΣEVR − 1\| | 0 |
| rank(X_c) | **279** = N − 1, as X_c **1** = 0 forces |
| σ₁, σ₂₈₀ | 78.4529, 1.3 × 10⁻¹³ (zero to machine precision) |
| total variance trace(C) | 112.3663 |
| ‖X − X̂‖_F / ‖X‖_F with k = 279 | 6.9 × 10⁻¹⁶ (training faces are reconstructed exactly) |
| ‖X − X̂‖_F / ‖X_c‖_F vs √(1 − cumulative(k)) | equal for every k tested (Eckart–Young) |

First explained-variance ratios (%): 19.63, 14.33, 7.00, 5.98, 5.15, 3.62, 2.71, 2.42, 1.99, 1.86.

Smallest k reaching 80 / 90 / 95 / 99 % of the variance: **29 / 66 / 114 / 212**.

## 7. Projection mathematics

- Training: Z_train = U_kᵀ X_c (`project_data`).
- New face: z = U_kᵀ(x − μ) with the **training** μ and U_k (`project_new_face`). x must have shape (d,) and μ shape (d, 1); other shapes are rejected, never silently reshaped.
- Reconstruction: x̂ = μ + U_k z (`reconstruct`).

## 8. Recognition mathematics

d_i = ‖z − z_i‖₂ (`euclidean_distances`), and ŷ = y_{i*} with i* = argmin_i d_i (`recognize`; on an exact tie, the smaller index wins). This is nearest neighbour in PCA space. No other model is involved.

## 9. Exact experimental results (seed 42, 280 train / 120 test)

| k | variance retained | accuracy | correct / total | mean NN distance | median NN distance | min | max |
|---|---|---|---|---|---|---|---|
| 5 | 52.1 % | 85.8 % | 103 / 120 | 1.498 | 1.370 | 0.320 | 4.966 |
| 10 | 64.7 % | 95.8 % | 115 / 120 | 2.190 | 2.059 | 0.460 | 5.484 |
| 20 | 75.4 % | 97.5 % | 117 / 120 | 2.870 | 2.618 | 0.712 | 7.468 |
| 30 | 80.9 % | **98.3 %** | 118 / 120 | 3.282 | 2.906 | 0.825 | 7.753 |
| 50 | 87.1 % | **98.3 %** | 118 / 120 | 3.795 | 3.348 | 1.001 | 8.337 |
| 66 | 90.1 % | 97.5 % | 117 / 120 | 4.107 | 3.700 | 1.121 | 8.547 |

- These are identical to the Phase 4 results. The result files were byte-compared after a full regeneration.
- Best: k = 30 and k = 50 tie. The rule fixed in advance (smallest k among ties) gives **k = 30**.
- At k = 30, the mean nearest-neighbour distance is 3.217 for correct predictions and 7.122 for wrong ones.

## 10. Confusion matrix

`outputs/figures/confusion_matrix_k30.png` (40 × 40, rows = true, columns = predicted). It contains 118 diagonal entries and 2 errors:

| test column | true | predicted | distance |
|---|---|---|---|
| 0 | s1 | s16 | 6.716 |
| 29 | s10 | s38 | 7.528 |

Test column 0 is also the fixed example in `recognize_single_face.py`, which is misrecognised at k = 66 too (distance 7.699). It was kept as the example deliberately, rather than swapped for a face that works.

## 11. Leakage prevention

- μ is computed from X_train only (`run_recognition.py` step 2; the custom demo's `train_model` receives only training columns).
- U and Σ come from the SVD of X_train − μ only.
- Test faces are centered with the training μ and projected with the training U_k, one at a time (`recognize_faces`).
- No test image is used to compute PCA. This is verified by `tests/test_recognition.py`:
  - `test_pca_uses_training_data_only`
  - `test_each_test_face_is_recognised_independently`: changing one test face leaves every other prediction bit-identical.
- Custom demo: `check_no_overlap` refuses any test file that is byte-identical to a training file, and `recognize` warns if the image is a training image.

## 12. Limitations

- One split of one small, controlled dataset: 40 people, frontal, plain background, similar scale. The 98.3 % is an estimate for this split. With 120 test faces one face is 0.83 percentage points, so the differences between k = 20 and k = 66 (1–2 faces) are not meaningful.
- Closed-set only: every test person is known. There is no unknown-person rejection.
- Eigenfaces depend on raw pixel alignment. Changes in lighting, pose, scale or background can dominate the leading components. There is no face detection or alignment, so custom photos must be cropped consistently by hand.
- Variance retained is a reconstruction criterion. It does not predict recognition accuracy: 81 % variance (k = 30) recognises as well as 87 % and better than 90 %.
- Distances are not probabilities, and they grow with k, so they cannot be compared across different k.

## 13. Reproducibility instructions (all verified)

```bash
python -m venv .venv
.venv\Scripts\activate                  # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt         # numpy, matplotlib, pillow, pytest
python -m pytest                        # 87 tests
python -m experiments.run_all           # downloads the dataset if needed; all results + figures
python demo.py sample                   # optional rehearsal data (4 AT&T people)
python demo.py --data-dir data/demo_sample train
python demo.py                          # interactive custom demo
```

Verified on 2026-10-05:
- **Development environment** (Python 3.13.7, numpy 2.3.3, matplotlib 3.10.7, Pillow 11.3.0): deleting `outputs/results` and `outputs/figures` and running `run_all` regenerates **byte-identical** result files and all 17 figures.
- **Fresh environment:** a fresh copy without the dataset, in a new virtual environment with only `requirements.txt`. pip chose numpy 2.5.3, matplotlib 3.11.2, Pillow 12.3.0 and pytest 9.1.1.
  - The checksum-verified download worked, and **87/87 tests passed with 0 skipped**.
  - `run_all` succeeded and all 17 figures were produced.
  - `dataset_report.txt`, `recognition_results.csv` and `recognition_summary.txt` were **byte-identical**.
  - `pca_spectrum.csv` was identical to 10 significant digits for all 279 meaningful components. Only machine-precision quantities differed: σ₂₈₀ (1.3161 vs 1.3167 × 10⁻¹³, a true zero) and the ~10⁻¹⁵ verification residuals in `pca_summary.txt`. Such last-bit rounding varies between NumPy/BLAS builds and is expected.
- **Tests before the dataset exists:** if `pytest` runs before the dataset is downloaded, the 3 real-data tests are skipped (84 passed, 3 skipped). Run `python -m experiments.run_all` (or `inspect_dataset`) first.

## 14. Known caveats and changes made during this audit

**Code changes (none alter the AT&T results; all result files byte-identical before and after):**
1. `discover_images` sorted digit-free folder names (e.g. "Alice", "Bob") by an all-zero key, so their order depended on the filesystem. It now uses a natural sort, which gives the same order for AT&T, and ignores hidden folders.
2. `load_image` now applies EXIF orientation, so phone photos are not sideways. This is a no-op for AT&T.
3. `train_test_split` now delegates to a new `split_indices`, so the custom demo can reuse the identical split. Same RNG and same order.
4. New `numerical_rank` (same tolerance as `numpy.linalg.matrix_rank`). `run_pca.py` uses it, and the demo uses it to cap k at rank(X_c).
5. The dataset zip's SHA-256 is now verified before extraction.
6. `run_pca.py` now also reports ‖X_c **1**‖, \|ΣEVR − 1\| and the full-rank reconstruction error.

**Caveats:**
- `pca_spectrum.csv` lists σ₂₈₀ ≈ 10⁻¹³. It is a rounding artefact of a true zero and must not be interpreted as a direction.
- The global Python on the development machine also has scikit-learn, TensorFlow and PyTorch installed. The project does not import them, and the fresh-environment run proves they are not needed.
- `recognize_single_face.py` shows a *misrecognised* face by default (test face 0). Use `--index 4` (or any other index) to show a correct case.
