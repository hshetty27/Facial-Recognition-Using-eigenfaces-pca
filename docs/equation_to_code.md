# Equation → Code

Every major equation of the project, the exact place in the code where it is computed, and the matrix shapes involved.
Line numbers refer to the current version of the files.

**Convention:** X ∈ ℝ^(d × N), with **one face per column**. For the AT&T experiment:

| Quantity | Value |
|---|---|
| image | 78 × 64 (h × w) |
| d = h·w | 4992 |
| N_train | 280 |
| N_test | 120 |
| rank(X_c) | 279 |
| k | one of 5, 10, 20, 30, 50, 66 |

| # | Equation | Meaning | File : line | Function | Key variables | Shapes |
|---|---|---|---|---|---|---|
| 0a | I_rc = p_rc / 255 | grey levels → real matrix in [0, 1] | `src/preprocessing.py:80` | `image_to_array` | `image` → array | (78, 64) |
| 0b | centre crop to H/W = h/w, then resize to h × w | common size, so every face is in the same ℝ^d, with equal scale in both directions (no stretching) | `src/preprocessing.py:40`, `:65` | `center_crop_to_aspect`, `resize_image` | `image_shape = IMAGE_SHAPE = (78, 64)` | any → (78, 64) |
| 1 | **x** = vec(I), x_{(r−1)w+c} = I_rc | image → vector (row-major) | `src/preprocessing.py:93` | `image_to_vector` (called by `preprocess_image`) | `image_array.reshape(-1)` | (78, 64) → (4992,) |
| 1b | X = [x_1 \| … \| x_N] | stack faces as columns | `src/preprocessing.py:133` | `build_data_matrix` | `np.column_stack(vectors)` | N × (4992,) → (4992, N) |
| 2 | **μ** = (1/N) Σ x_i = (1/N) X **1** | mean face | `src/pca.py:37` | `compute_mean_face(X)` | `ones` (N, 1), `(X @ ones) / N` | (4992, 280) → (4992, 1) |
| 3 | **X_c** = X − μ**1**ᵀ | centering; each row then has mean 0, X_c **1** = 0 | `src/pca.py:53` | `center_data(X, mean)` | `X - mean` (broadcast over columns; a mean of any other shape is rejected) | (4992, 280) |
| 4 | **X_c = U Σ Vᵀ** (thin) | SVD; columns of U are the principal directions | `src/pca.py:73` | `compute_svd(X_centered)` | `np.linalg.svd(X_centered, full_matrices=False)` → `U, S, Vt` | U (4992, 280), S (280,), Vt (280, 280) |
| 4b | rank(X_c) = #{σ_i > σ₁ · max(d, N) · ε} | numerical rank (≤ N − 1 = 279) | `src/pca.py:145` | `numerical_rank(S, shape)` | `tolerance` | scalar → 279 |
| 5 | C = X_cX_cᵀ/(N−1) = U[Σ²/(N−1)]Uᵀ | covariance; **never formed** (24.9 M entries) | (tests only) | `tests/test_pca.py::test_columns_of_U_are_eigenvectors_of_covariance` | `C = X_c @ X_c.T / (N - 1)` on a small d = 5 matrix | (d, d) |
| 5b | **λ_i = σ_i² / (N − 1)**, C u_i = λ_i u_i | eigenvalues = variance along u_i | `src/pca.py:92` | `eigenvalues_from_singular_values(S, N)` | `singular_values**2 / (n_samples - 1)` | (280,) |
| 6 | **EVR_i = σ_i² / Σⱼ σ_j²**, cumulative(k) = Σᵢ≤ₖ EVR_i | explained variance | `src/pca.py:111–112` | `explained_variance_ratio(S)` | `evr`, `np.cumsum(evr)` | (280,), (280,) |
| 6b | k(τ) = min{k : cumulative(k) ≥ τ} | smallest k reaching a variance threshold | `src/pca.py` | `choose_k(cumulative, threshold)` | — | scalar |
| 6c | U_k = [u_1 \| … \| u_k], U_kᵀU_k = I_k | keep the first k directions (eigenfaces) | `src/pca.py:162` | `get_principal_components(U, k)` | `U[:, :k]` | (4992, k) |
| 7a | **Z_train = U_kᵀ X_c** | PCA coordinates of all training faces | `src/projection.py:28` | `project_data(X_centered, U_k)` | `U_k.T @ X_centered` | (k, 280) |
| 7b | **z = U_kᵀ (x − μ)** | PCA coordinates of one new face | `src/projection.py:51–52` | `project_new_face(face_vector, mean_face, principal_components)` | `x_c = face_vector - mean_face[:, 0]`, `principal_components.T @ x_c` | x (4992,), μ (4992, 1) → z (k,) |
| 8 | **x̂ = μ + U_k z**, X̂ = U_kZ + μ | reconstruction from k coordinates | `src/projection.py:68` | `reconstruct(Z, mu, U_k)` | `U_k @ Z + mu` | (k, N) → (4992, N) |
| 8b | e_i = ‖x_i − x̂_i‖₂; ‖X − X̂‖_F / ‖X_c‖_F | reconstruction errors | `src/projection.py:84`, `src/projection.py` | `reconstruction_error`, `relative_reconstruction_error` | — | (N,), scalar |
| 9 | **d_i = ‖z − z_i‖₂** | Euclidean distance in PCA space | `src/recognition.py:29–30` | `euclidean_distances(query, database)` | `differences = database - query[:, np.newaxis]`, `np.sqrt(np.sum(differences**2, axis=0))` | z (k,), Z_train (k, 280) → (280,) |
| 10 | **ŷ = y_{i*}, i* = argmin_i d_i** | nearest-neighbour identity | `src/recognition.py:49–50` | `recognize(query_projection, training_projections, labels)` | `nearest = int(np.argmin(distances))` → `(label, index, distance)` | scalar |
| 10b | ŷ_j for every test column | apply 7b + 9 + 10 to each test face separately | `src/recognition.py` | `recognize_faces(X_test, mean_face, principal_components, training_projections, labels)` | loop over columns of X_test | (4992, 120) → (120,) |
| 11 | accuracy = #{ŷ_j = y_j} / M | fraction correct | `src/recognition.py:91` | `accuracy(predictions, true_labels)` | `np.mean(predictions == true_labels)` | scalar |
| 11b | M[a, b] = #{true a, predicted b} | confusion matrix | `src/recognition.py` | `confusion_matrix(true_labels, predictions, classes)` | `counts` | (40, 40) |

## Where the pipeline is assembled

| Step | AT&T experiment | Custom demo |
|---|---|---|
| images → X, labels | `src/data_loader.py::load_dataset` | `src/custom_demo.py::load_labelled_folder` (same `load_image` + `preprocess_image` + `build_data_matrix`) |
| train/test split | `src/data_loader.py::train_test_split` → `split_indices` (seed 42) | `src/custom_demo.py::prepare_custom_split` → the same `split_indices`, plus a check that no test file is byte-identical to a training file |
| μ, X_c, SVD, U_k, Z_train | `experiments/run_recognition.py` (step by step) | `src/custom_demo.py::train_model` (the same `pca.py` / `projection.py` functions; k reduced to the rank if larger) |
| new face → identity | `src/recognition.py::recognize_faces` | `src/custom_demo.py::recognize_image` (`project_new_face` → `euclidean_distances` → `recognize`) |

## Shape walk-through (AT&T, k = 30)

```
image (112 x 92 PGM) --crop (no-op for AT&T)--> --resize--> (78, 64) --/255--> I (78, 64) --reshape--> x (4992,)
400 vectors --column_stack--> X (4992, 400) --split seed 42--> X_train (4992, 280), X_test (4992, 120)
X_train --mean--> mu (4992, 1) --subtract--> X_c (4992, 280)
X_c --svd--> U (4992, 280), S (280,), Vt (280, 280);  rank 279, sigma_280 ~ 1e-13
U --[:, :30]--> U_k (4992, 30) --U_k^T X_c--> Z_train (30, 280)
x_test (4992,) --minus mu--> (4992,) --U_k^T--> z (30,) --distances--> (280,) --argmin--> label
```
