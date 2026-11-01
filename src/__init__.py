"""
EigenFaces: Facial Recognition Through PCA, Eigenvectors and Projection.

Matrix convention used throughout the project
---------------------------------------------
- d : number of pixels per image (78 * 64 = 4992 in this project)
- N : number of face images
- k : number of principal components kept (k <= min(d, N))

The data matrix X has shape (d, N): each COLUMN is one face vector x_i.
With this orientation the PCA projection is written exactly as

    Z = U_k^T X_centered        (shape (k, N))
"""
