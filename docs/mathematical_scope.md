# Mathematical Scope

This document defines the mathematics of the project, using the notation of the code.

## Notation and convention

| Symbol | Meaning | Shape / value |
|---|---|---|
| h, w | image height and width in pixels | 78, 64 |
| d = h·w | number of pixels = dimension of face space | 4992 |
| N | number of images in the matrix being analysed | N_train = 280 |
| k | number of principal components kept | 1 ≤ k ≤ rank(X_c) |
| x_i | i-th face as a vector | d × 1 |
| X = [x_1 \| … \| x_N] | data matrix, **one face per column** | d × N |
| **1** | vector of ones | N × 1 |
| μ | mean face | d × 1 |
| X_c | centered data matrix | d × N |
| C | sample covariance matrix (never formed in code) | d × d |
| U, Σ, Vᵀ | thin SVD of X_c, r = min(d, N) | d × r, r × r, r × N |
| σ_i, λ_i | singular values of X_c, eigenvalues of C | — |
| U_k | first k columns of U | d × k |
| Z | coordinates of faces along u_1, …, u_k | k × N |

---

## From Face Image to Data Matrix

This section defines the input to PCA.

**Step 1: an image is a matrix.**
Each stored AT&T image is 112 × 92 pixels (height × width) with 8-bit grey levels. It is resized to 78 × 64 pixels. This keeps the aspect ratio (78/64 = 1.219 ≈ 112/92 = 1.217), and both directions are scaled by the same factor, 0.696. Nothing is cropped. (Images of other shapes, such as custom-demo photos, are first centre-cropped to 78 : 64 so that they too are scaled equally in both directions; for AT&T this crop does nothing.) The grey levels are then divided by 255:

I ∈ ℝ^(78×64),  I_rc = p_rc / 255 ∈ [0, 1]

The division by 255 is a scalar multiplication. It only keeps the numbers in a convenient range and changes no geometric relationship between faces.

**Step 2: flattening turns the matrix into a vector.**
The 78 rows of I are laid end to end (row-major order):

x = (I₁,₁, …, I₁,₆₄, I₂,₁, …, I₇₈,₆₄)ᵀ ∈ ℝ⁴⁹⁹²,  x_{(r−1)·64 + c} = I_rc

So a 78 × 64 grayscale image becomes 4992 scalar pixel values, i.e. one vector in ℝ⁴⁹⁹². The map is a bijection: reshaping x into 78 × 64 recovers I exactly.

**Step 3: N vectors are assembled column-wise.**

X = [x_1 x_2 … x_N] ∈ ℝ^(4992×N)

- Row p of X holds the intensity of pixel p across all N faces.
- Column j of X is face j: one point in ℝ⁴⁹⁹².

**Labels are kept separate.** A label array y = (y_1, …, y_N), where y_j is the person shown in column j. X contains pixel intensities only.

| Quantity | Value |
|---|---|
| d | 78 · 64 = 4992 |
| N (all images) | 400 (40 people × 10 images) |
| X | 4992 × 400 |
| X_train / X_test | 4992 × 280 / 4992 × 120 (7 / 3 images per person, seed 42) |
| pixel values | [0, 1] (observed range [0.0196, 0.9098]) |

---

## Principal Component Analysis of Facial Data

Everything below is computed from the **training** matrix X_train ∈ ℝ^(4992×280) only. In this section, X means X_train and N = 280.

### 1. Mean face

μ = (1/N) Σᵢ₌₁ᴺ x_i = (1/N) X **1** ∈ ℝ^(d×1)

Entry p of μ is the average intensity of pixel p over the N training faces. μ is the centre of the cloud of N points in ℝ^d. It is stored as a d × 1 column so that it can be subtracted from every column of X.

*Code:* `compute_mean_face(X)` computes `(X @ ones) / N` → shape (4992, 1).

### 2. Centering

x_c,i = x_i − μ,  X_c = X − μ **1**ᵀ ∈ ℝ^(d×N)

In NumPy, `X - mu` with `mu` of shape (d, 1) *broadcasts*: the column μ is subtracted from each of the N columns. No transposition is involved, and `center_data` rejects a mean of any other shape.

Consequences:
- every row of X_c has mean 0 (each pixel is centered);
- X_c **1** = Σᵢ x_c,i = 0, so the columns of X_c are linearly dependent.

Centering moves the origin to the mean face, so that PCA describes how faces *vary* around the average face. Without centering, the first direction would mostly point towards μ itself.

### 3. Covariance matrix

C = (1/(N−1)) X_c X_cᵀ ∈ ℝ^(d×d)

- C_pp is the sample variance of pixel p.
- C_pq is the sample covariance of pixels p and q.
- C is symmetric and positive semi-definite (vᵀCv = ‖X_cᵀv‖²/(N−1) ≥ 0).

For a unit vector u, the projections uᵀx_c,i have sample variance

Var(u) = (1/(N−1)) Σᵢ (uᵀx_c,i)² = uᵀ C u.

PCA asks for the unit vectors u that maximise uᵀCu. By the spectral theorem, C = Σ λ_i u_i u_iᵀ with orthonormal eigenvectors u_i. The maximum of uᵀCu over ‖u‖ = 1 is λ₁, attained at u = u₁. The next maximum, over u ⟂ u₁, is λ₂ at u₂, and so on. **The principal directions are the eigenvectors of C, and the variances along them are the eigenvalues.**

C itself is never formed. It would be a 4992 × 4992 matrix with about 24.9 million entries, and its rank is at most 279. The SVD of X_c gives the same eigenvectors and eigenvalues.

### 4. Singular Value Decomposition

X_c = U Σ Vᵀ  (thin SVD, `numpy.linalg.svd(X_c, full_matrices=False)`)

| Factor | Shape (general) | Shape (our data) | Property |
|---|---|---|---|
| U | d × r | 4992 × 280 | orthonormal columns: UᵀU = I_r |
| Σ = diag(σ₁, …, σ_r) | r singular values | 280 | σ₁ ≥ σ₂ ≥ … ≥ σ_r ≥ 0 |
| Vᵀ | r × N | 280 × 280 | orthonormal rows: VᵀV = I_r |

with r = min(d, N) = min(4992, 280) = 280.

**Why the rank cannot exceed min(d, N):** the rank of a matrix is the dimension of its column space. That space is spanned by N columns, so its dimension is at most N. It is also a subspace of ℝ^d, so its dimension is at most d. Hence rank(X_c) ≤ min(d, N), and at most r = min(d, N) singular values can be non-zero.

**One more dimension is lost by centering.** X_c **1** = 0 is a non-trivial linear dependency among the columns, so rank(X_c) ≤ N − 1 = 279. On the real data, σ₂₈₀ ≈ 1.3 × 10⁻¹³, which is zero to machine precision, and the numerical rank is exactly 279.

### 5. Eigenvalue / eigenvector relationship

Substituting the SVD and using VᵀV = I:

X_c X_cᵀ = U Σ Vᵀ V Σ Uᵀ = U Σ² Uᵀ

C = (1/(N−1)) X_c X_cᵀ = U [Σ²/(N−1)] Uᵀ

Multiplying on the right by u_i, and using Uᵀu_i = e_i:

C u_i = (σ_i² / (N−1)) u_i,  i.e.  A v = λ v with A = C, v = u_i, λ = λ_i.

Therefore:
- **the columns of U are eigenvectors of C**, and
- **λ_i = σ_i² / (N − 1)** is the variance of the training faces along u_i.

C also has d − r = 4712 further eigenvalues equal to 0. Their eigenvectors span directions in which the training faces do not vary at all.

*Code:* `eigenvalues_from_singular_values(S, N)` computes `S**2 / (N - 1)`. `numpy.linalg.eig` is not used. The tests only use an eigensolver as an independent cross-check, on a small matrix.

*Verified on the real data:* ‖C U₁₀ − U₁₀ Λ₁₀‖ / ‖U₁₀ Λ₁₀‖ ≈ 1.6 × 10⁻¹⁵. Here C was applied as X_c(X_cᵀu)/(N−1), without forming C.

### 6. Principal components

U_k = [u_1 | u_2 | … | u_k] ∈ ℝ^(d×k),  U_kᵀ U_k = I_k

- u_1 = first principal direction: the unit direction of maximum variance λ₁.
- u_2 = second principal direction: maximum remaining variance, orthogonal to u_1.
- In general, u_i maximises the variance subject to u_i ⟂ u_1, …, u_{i−1}.

Each u_i ∈ ℝ⁴⁹⁹² can be reshaped to 78 × 64 and displayed. These images are the **eigenfaces**. They are *not* faces of real people. They are unit vectors whose entries can be positive or negative: patterns of pixel changes away from the mean face. Two points about displaying them:
- Each eigenface is linearly rescaled to the full grey range for display only. The vectors u_i are not modified.
- The sign is arbitrary: if u_i is an eigenvector, so is −u_i. An eigenface and its photographic negative carry the same information.

### 7. Explained variance

The total variance of the data is

trace(C) = Σᵢ λ_i = Σ_p Var(pixel p) = ‖X_c‖²_F / (N−1).

The fraction of it along u_i, and the fraction kept by the first k directions, are

EVR_i = λ_i / Σⱼ λ_j = σ_i² / Σⱼ σ_j²,  cumulative(k) = Σᵢ₌₁ᵏ EVR_i.

The factor 1/(N−1) cancels. Σᵢ EVR_i = 1, and cumulative(k) is non-decreasing in k.

**Choosing k:** for a threshold τ ∈ (0, 1], k(τ) = min{ k : cumulative(k) ≥ τ }. k is not chosen from variance alone: recognition is evaluated for several k (see below).

*Results (training faces):*

| k | 5 | 10 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| cumulative variance | 52.1 % | 64.7 % | 75.4 % | 80.9 % | 87.1 % | 94.0 % |

| threshold τ | 80 % | 90 % | 95 % | 99 % |
|---|---|---|---|---|
| k(τ) | 29 | 66 | 114 | 212 |

### 8. Projection

Z = U_kᵀ X_c ∈ ℝ^(k×N),  z_ji = u_jᵀ (x_i − μ)

Because the u_j are orthonormal, the j-th coordinate of a face is simply its dot product with u_j. Each face is now described by k numbers instead of d = 4992. This is the **dimensionality reduction**.

Using the SVD, Z = U_kᵀ U Σ Vᵀ = Σ_k V_kᵀ. So the coordinates along u_j have sample variance σ_j²/(N−1) = λ_j.

A new face x is projected in the same way, z = U_kᵀ(x − μ), with the **training** μ and U_k (see the next section).

### 9. Reconstruction

x̂ = U_k z + μ = U_k U_kᵀ (x − μ) + μ,  X̂ = U_k Z + μ **1**ᵀ

P_k = U_k U_kᵀ is the orthogonal projection matrix onto span(u_1, …, u_k) (P_k² = P_k, P_kᵀ = P_k). So x̂ is the point of the affine subspace μ + span(U_k) closest to x.

**Error metrics used:**
- For one face: the Euclidean norm e_i = ‖x_i − x̂_i‖₂, which is the length of the part of x_i − μ orthogonal to span(U_k).
- For a whole matrix: the normalised Frobenius error ‖X − X̂‖_F / ‖X_c‖_F. The denominator is the centered data, so 1 means "no better than the mean face" and 0 means perfect.

**Eckart–Young:** for the training data, ‖X_c − U_kU_kᵀX_c‖²_F = Σ_{i>k} σ_i², which is the minimum possible over all rank-k approximations. Hence

‖X − X̂‖_F / ‖X_c‖_F = √(1 − cumulative(k)).

This identity holds exactly on the real data. For example, k = 50 gives 0.3592 both ways.

With k = rank(X_c) = 279, every *training* face is reconstructed exactly, because X_c lies entirely in span(u_1, …, u_279). This is expected, not a sign of good generalisation.

---

## Facial Recognition in PCA Space

### The method

PCA is learned once, from the training faces only (previous section): μ, U and, for a chosen k, U_k.

**Training (the "database" of known faces):**

Z_train = U_kᵀ (X_train − μ) = U_kᵀ X_c ∈ ℝ^(k × N_train)

Column i, z_i ∈ ℝᵏ, holds the PCA coordinates of training face i. Its label y_i is stored separately.

**Test face** x_test ∈ ℝ^d, never used before this point:

z_test = U_kᵀ (x_test − μ) ∈ ℝᵏ  (training μ and training U_k)

*Code:* `project_new_face(x, mu, U_k)` takes x of shape (d,), μ of shape (d, 1) and U_k of shape (d, k), and returns z of shape (k,). Any other input shape is rejected rather than reshaped.

**Distance** to every known face:

d_i = ‖z_test − z_i‖₂ = √( Σⱼ₌₁ᵏ (z_test,j − z_ji)² ),  i = 1, …, N_train

**Prediction** (nearest neighbour):

i* = argmin_i d_i,  identity = y_{i*}

If two distances are exactly equal, the smaller index wins. There is no classifier and no learned parameter beyond PCA itself.

### Geometric meaning

Each face, known or new, becomes a **point in k-dimensional PCA space**. Its coordinates are its components along the orthonormal eigenfaces u_1, …, u_k. Recognition asks one question:

> *Which known face is closest to this new point?*

This is the link between projection and recognition. Projection removes the d − k directions in which training faces vary least. It keeps the k directions of largest variation, which are where differences between faces show up most strongly.

Because U_k has orthonormal columns, the distance in PCA space equals the distance between the *orthogonal projections* of the two centered faces onto span(U_k):

‖z − z_i‖₂ = ‖U_kU_kᵀ(x − μ) − U_kU_kᵀ(x_i − μ)‖₂.

So recognition in PCA space is pixel-space nearest neighbour after the low-variance directions have been projected away. With k = rank(X_c), distances between training faces are exactly their pixel-space distances (verified in the tests).

**Distances grow with k.** d_i² is a sum of k non-negative terms, so each distance can only increase when k increases. Distance values are therefore only comparable for the same k.

### No test-data leakage

- μ is computed from X_train only.
- U and Σ are computed from the SVD of X_train − μ only.
- Each test face is centered with the **training** μ and projected with the **training** U_k.
- No test image is used to compute the PCA. Each test face is processed independently.

The tests verify this in three ways:
- μ equals the training mean and differs from the mean of train and test combined.
- U comes from the training SVD.
- Changing one test face leaves the predictions and distances of all other test faces bit-for-bit unchanged. If the PCA had used the test set, that change would have moved μ and U.

### The PC1–PC2 plot is not a new method

`pca_space_pc1_pc2.png` plots, for every training face, (z_1i, z_2i) = (u_1ᵀ(x_i − μ), u_2ᵀ(x_i − μ)): the first two rows of Z_train, which PCA has already computed. It is **not** another dimensionality-reduction algorithm (such as t-SNE). It is a view of 2 of the k coordinates used for recognition. Identity labels are used only to colour the points.

Some people form tight groups even in these two coordinates (s1, s2, s5), one splits into two groups (s6), and others overlap (s3, s4). This is why recognition uses k ≫ 2 coordinates.

### Reconstruction quality vs recognition accuracy

These are different questions and must not be confused.

- **Explained variance / reconstruction error** measure how well the *training* faces are approximated by k components: ‖X − X̂‖_F / ‖X_c‖_F = √(1 − cumulative(k)). This is a property of the training data alone.
- **Recognition accuracy** measures how often a *new, unseen* face lands closest to a training face of the same person. It must be measured on held-out test faces.

High explained variance means the components retain most of the variation needed to *reconstruct* faces. It does **not** guarantee equally high recognition accuracy:
- some high-variance directions describe lighting or pose, not identity;
- small directions can still help separate two similar people.

The measured results (below) show that 30 components (80.9 % of the variance) already recognise as well as 50, and slightly better than 66 (90.1 %).

### Results (fixed split, seed 42; 280 training / 120 test faces)

| k | variance retained | accuracy | correct | mean NN distance | median NN distance |
|---|---|---|---|---|---|
| 5 | 52.1 % | 85.8 % | 103 / 120 | 1.498 | 1.370 |
| 10 | 64.7 % | 95.8 % | 115 / 120 | 2.190 | 2.059 |
| 20 | 75.4 % | 97.5 % | 117 / 120 | 2.870 | 2.618 |
| 30 | 80.9 % | **98.3 %** | 118 / 120 | 3.282 | 2.906 |
| 50 | 87.1 % | **98.3 %** | 118 / 120 | 3.795 | 3.348 |
| 66 | 90.1 % | 97.5 % | 117 / 120 | 4.107 | 3.700 |

- Best: k = 30 and k = 50 tie at 98.3 %. The rule fixed in advance (smallest k among ties) reports **k = 30**.
- Accuracy rises steeply up to k ≈ 10–20, then is flat. It is not monotonic: k = 66 is one face worse than k = 50.
- With 120 test faces, one face is 0.83 percentage points. The differences between k = 20 and k = 66 are 1–2 faces and should not be over-interpreted.

### Limitations

- A single fixed split of one small dataset (40 people, 10 images each). The accuracy is an estimate for this split, not a general property of the method.
- Closed-set only: every test face belongs to one of the 40 known people. Rejecting unknown people (open-set recognition) is not addressed.
- The AT&T images are controlled: frontal, plain background, similar scale. Eigenfaces are known to be sensitive to lighting, pose and alignment, which this dataset mostly avoids.

---

## Concept summary

| Concept | Mathematical expression | Role in the project |
|---|---|---|
| Vectors | x_i ∈ ℝ⁴⁹⁹² | each face is one point in pixel space |
| Matrices | X ∈ ℝ^(4992×N) | all faces, one per column |
| Mean-centering | μ = (1/N) X**1**, X_c = X − μ**1**ᵀ | describe variation around the average face |
| Variance / covariance | C = X_cX_cᵀ/(N−1), Var(u) = uᵀCu | what PCA maximises |
| Eigenvalues | λ_i = σ_i²/(N−1) | variance along each direction; used to choose k |
| Eigenvectors | C u_i = λ_i u_i | principal directions = eigenfaces |
| Orthogonality | U_kᵀU_k = I_k | projection reduces to dot products |
| PCA | maximise uᵀCu subject to ‖u‖ = 1, u ⟂ previous directions | finds the low-dimensional face subspace |
| SVD | X_c = UΣVᵀ | computes PCA without forming the d × d matrix C |
| Projection | Z = U_kᵀ X_c | k coordinates per face |
| Dimensionality reduction | ℝ⁴⁹⁹² → ℝᵏ | e.g. k = 66 keeps 90 % of the variance |
| Reconstruction | x̂ = U_kU_kᵀ(x − μ) + μ | shows what is kept and lost |
| Euclidean distance | ‖z − z_i‖₂ | comparing projected faces; nearest neighbour = recognised identity |

## Out of scope

Neural networks, deep learning, pretrained embeddings, and library PCA implementations (e.g. `sklearn.decomposition.PCA`) are deliberately excluded.
