"""
Turning images into vectors and building the data matrix X.

Pipeline for one image (orientation is corrected earlier, in data_loader.load_image):

    Pillow image --to_grayscale--> mode "L"
                 --center_crop_to_aspect--> height/width = h/w, nothing stretched
                 --resize_image--> h x w pixels
                 --image_to_array--> I in R^(h x w), values in [0, 1]
                 --image_to_vector--> x in R^d,  d = h * w

Then N vectors are stacked as columns:  X = [x_1 | x_2 | ... | x_N] in R^(d x N).

Shapes are always written (height, width), as NumPy does. Pillow's resize
expects (width, height); the conversion happens in one place only (resize_image).
"""

import numpy as np
from PIL import Image

MAX_INTENSITY = 255.0  # 8-bit grayscale

# Project image size (height, width). AT&T originals are 112 x 92 (h x w);
# 78 / 64 = 1.219 is close to 112 / 92 = 1.217, so the aspect ratio is preserved.
# d = 78 * 64 = 4992.
IMAGE_SHAPE = (78, 64)


def to_grayscale(image: Image.Image) -> Image.Image:
    """
    Convert an image to 8-bit grayscale (Pillow mode "L").

    For colour images Pillow uses the ITU-R 601-2 luma transform
        L = 0.299 R + 0.587 G + 0.114 B.
    Images that are already grayscale (like the AT&T faces) are unchanged.
    """
    return image if image.mode == "L" else image.convert("L")


def center_crop_to_aspect(image: Image.Image, image_shape: tuple[int, int]) -> Image.Image:
    """
    Crop symmetrically around the centre so that height / width matches image_shape.

    Input:  Pillow image of size W x H (width x height); image_shape = (h, w).
    Output: the largest centred crop with H' / W' = h / w (to the nearest pixel).

    Too tall  (H / W > h / w): keep all W columns, keep H' = round(W * h / w) rows.
    Too wide  (H / W < h / w): keep all H rows,    keep W' = round(H * w / h) columns.
    The crop is then scaled by the SAME factor in both directions by resize_image,
    so faces are not stretched. Example: a 448 x 672 (w x h) phone photo, target
    78 / 64 = 1.21875, keeps rows 63 .. 608 (546 rows, 546 / 448 = 1.21875).
    AT&T images (92 x 112) already have the ratio to within half a pixel: no crop.
    """
    height, width = image_shape
    W, H = image.size
    crop_w = min(W, round(H * width / height))
    crop_h = min(H, round(W * height / width))
    left = (W - crop_w) // 2
    top = (H - crop_h) // 2
    if (crop_w, crop_h) == (W, H):
        return image
    return image.crop((left, top, left + crop_w, top + crop_h))


def resize_image(image: Image.Image, image_shape: tuple[int, int]) -> Image.Image:
    """
    Resize to a common size so every face lives in the same space R^d.

    Input:  Pillow image of any size; image_shape = (height, width).
    Output: Pillow image with `height` rows and `width` columns.

    Bilinear interpolation is used. This function does NOT preserve the aspect ratio
    by itself; preprocess_image calls center_crop_to_aspect first, so that the scale
    factors h/H and w/W are equal. For AT&T images (112 x 92, h x w) both are 0.696.
    """
    height, width = image_shape
    return image.resize((width, height), Image.Resampling.BILINEAR)


def image_to_array(image: Image.Image) -> np.ndarray:
    """
    Convert a grayscale image to a real matrix with entries in [0, 1].

    Input:  Pillow image, mode "L", values in {0, 1, ..., 255}.
    Output: I of shape (height, width), dtype float64, I = pixels / 255.

    The division by 255 is a scalar multiplication, used only for numerical
    convenience; it changes no geometric relationship between faces.
    """
    return np.asarray(image, dtype=np.float64) / MAX_INTENSITY


def image_to_vector(image_array: np.ndarray) -> np.ndarray:
    """
    Flatten an image matrix into a vector, row by row (row-major order).

    Input:  I of shape (h, w).
    Output: x of shape (d,), d = h * w, with x[r * w + c] = I[r, c].
            For 78 x 64: x in R^4992.
    """
    return image_array.reshape(-1)


def vector_to_image(vector: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """
    Inverse of image_to_vector: reshape x in R^d back into an (h, w) matrix.

    Used to check that flattening loses nothing, and later to display vectors as images.
    """
    height, width = image_shape
    if vector.size != height * width:
        raise ValueError(f"Vector of length {vector.size} cannot be shaped into {image_shape}")
    return vector.reshape(height, width)


def preprocess_image(image: Image.Image, image_shape: tuple[int, int] = IMAGE_SHAPE) -> np.ndarray:
    """
    Full pipeline for one face: grayscale -> centre crop to h/w -> resize -> [0, 1] values -> flatten.

    Input:  Pillow image, already upright (data_loader.load_image applies EXIF orientation);
            image_shape = (height, width).
    Output: x of shape (height * width,).

    This is the ONLY image -> vector function: training, held-out test, and new images
    all go through it, so they all live in the same R^d with the same geometry.
    """
    gray = to_grayscale(image)
    cropped = center_crop_to_aspect(gray, image_shape)
    resized = resize_image(cropped, image_shape)
    return image_to_vector(image_to_array(resized))


def build_data_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    """
    Stack N face vectors as the COLUMNS of the data matrix.

    Input:  list of N vectors, each of shape (d,).
    Output: X of shape (d, N), with X[:, j] = vectors[j].
    """
    return np.column_stack(vectors)


def verify_data_matrix(X: np.ndarray, labels: np.ndarray, image_shape: tuple[int, int]) -> dict:
    """
    Check that X and labels have the agreed shapes, and return a summary.

    Checks:  X.shape[0] == height * width   (one row per pixel)
             X.shape[1] == len(labels)      (one column per labelled face)
             labels.shape == (N,)
             all pixel values in [0, 1]
    Raises ValueError if any check fails.
    """
    height, width = image_shape
    d = height * width
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if X.shape[0] != d:
        raise ValueError(f"X has {X.shape[0]} rows, expected d = {height} x {width} = {d}")
    if labels.ndim != 1 or X.shape[1] != labels.shape[0]:
        raise ValueError(f"X has {X.shape[1]} columns but labels has shape {labels.shape}")
    if X.min() < 0.0 or X.max() > 1.0:
        raise ValueError(f"Pixel values outside [0, 1]: [{X.min()}, {X.max()}]")

    people, counts = np.unique(labels, return_counts=True)
    return {
        "image_shape": image_shape,
        "flattened_dimension": d,
        "n_images": X.shape[1],
        "n_people": len(people),
        "images_per_person": sorted(set(counts.tolist())),
        "data_matrix_shape": X.shape,
        "labels_shape": labels.shape,
        "pixel_min": float(X.min()),
        "pixel_max": float(X.max()),
        "dtype": str(X.dtype),
    }


def format_report(summary: dict) -> str:
    """Human-readable version of verify_data_matrix's summary."""
    h, w = summary["image_shape"]
    return "\n".join([
        f"Image dimensions:     {h} x {w}",
        f"Flattened dimension:  {summary['flattened_dimension']}",
        f"Number of images N:   {summary['n_images']}",
        f"Number of people:     {summary['n_people']}",
        f"Images per person:    {summary['images_per_person']}",
        f"Data matrix shape:    {summary['data_matrix_shape']}",
        f"Labels shape:         {summary['labels_shape']}",
        f"Pixel value range:    [{summary['pixel_min']:.4f}, {summary['pixel_max']:.4f}]",
        f"dtype:                {summary['dtype']}",
    ])
