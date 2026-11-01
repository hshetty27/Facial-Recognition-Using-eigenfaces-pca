"""
Before / after figure for the aspect-ratio fix in preprocess_image.

For each photo: the upright original, the OLD direct resize to 78 x 64 (stretched),
the centre-crop region on the original, and the NEW crop + resize to 78 x 64.
All panels are drawn with equal pixel aspect, so stretching is visible as it really is.

Run from the project root:
    python -m experiments.compare_preprocessing
    python -m experiments.compare_preprocessing demo_data/Friend_A/01.jpg mystery/Person_1_extra.jpg
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.data_loader import PROJECT_ROOT, load_image
from src.preprocessing import (
    IMAGE_SHAPE, center_crop_to_aspect, image_to_array, preprocess_image, resize_image, to_grayscale,
    vector_to_image,
)

DEFAULT_IMAGES = ["demo_data/Friend_A/01.jpg", "mystery/Person_1_extra.jpg"]
SAVE_PATH = PROJECT_ROOT / "outputs" / "demo" / "preprocessing_before_after.png"


def main(image_paths: list[str]) -> None:
    h, w = IMAGE_SHAPE
    fig, axes = plt.subplots(len(image_paths), 3, figsize=(10, 4.8 * len(image_paths)), squeeze=False)
    for row, name in zip(axes, image_paths):
        gray = to_grayscale(load_image(PROJECT_ROOT / name))
        W, H = gray.size
        old = image_to_array(resize_image(gray, IMAGE_SHAPE))           # previous method: direct resize
        new = vector_to_image(preprocess_image(gray, IMAGE_SHAPE), IMAGE_SHAPE)
        crop = center_crop_to_aspect(gray, IMAGE_SHAPE)
        left, top = (W - crop.width) // 2, (H - crop.height) // 2

        row[0].imshow(gray, cmap="gray")
        row[0].add_patch(Rectangle((left - 0.5, top - 0.5), crop.width, crop.height,
                                   fill=False, edgecolor="red", linewidth=2))
        row[0].set_title(f"BEFORE: {name}\n{W} x {H} (w x h), H/W = {H / W:.3f}\n"
                         f"red box = centre crop {crop.width} x {crop.height}", fontsize=9)
        row[1].imshow(old, cmap="gray", vmin=0, vmax=1)
        row[1].set_title(f"OLD: direct resize to {w} x {h}\nx-scale {w / W:.3f}, y-scale {h / H:.3f}"
                         f" -> stretched", fontsize=9)
        row[2].imshow(new, cmap="gray", vmin=0, vmax=1)
        row[2].set_title(f"NEW: centre crop + resize to {w} x {h}\nx-scale {w / crop.width:.3f}, "
                         f"y-scale {h / crop.height:.3f} -> proportions kept", fontsize=9)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"Aspect ratio: target h/w = {h}/{w} = {h / w:.5f}", fontsize=11)
    fig.tight_layout()
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_PATH, dpi=120)
    print(f"figure saved: {SAVE_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_IMAGES)
