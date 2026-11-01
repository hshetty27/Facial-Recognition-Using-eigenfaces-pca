# data/raw

Original face images: **AT&T Database of Faces** (formerly ORL), Olivetti Research Laboratory, Cambridge.

- Source: https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip (3.8 MB, no login)
- SHA-256 of the downloaded zip: `509d763bd2276aa054d260d7f1000e8be212738134f2f83d8ff13bc2adf1c8ec`
- Layout after extraction: `att_faces/s1/1.pgm` … `att_faces/s40/10.pgm`
  (40 people × 10 images, 92 × 112 pixels, 8-bit grayscale)

To download it, run from the project root:

```bash
python -m experiments.inspect_dataset
```

(or call `src.data_loader.download_att_faces()`). Nothing is re-downloaded if the files already exist.

These files are ignored by git (see `.gitignore`), so they are not committed.
When using the images, credit the Olivetti Research Laboratory.
