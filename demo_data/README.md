# demo_data

Put the photos for the custom demo here: **one sub-folder per person**. The folder name
becomes the identity label, and any number of people is allowed.

```
demo_data/
├── Alice/
│   ├── 01.jpg
│   ├── 02.jpg
│   └── ...
├── Bob/
└── ...
```

- Use at least 6 photos per person. 10 or more is better.
- Crop each photo to the **head**, as in the AT&T images: face centred, upright and frontal, in a
  portrait frame with height : width ≈ 1.22 (for example 640 × 780 pixels).
- Use similar lighting and a plain background for everyone.
- Formats: .jpg .jpeg .png .bmp .tif .tiff .webp .pgm. HEIC (iPhone) is **not** supported, so
  export those photos as JPG.

**Privacy:** only use photos of people who have agreed. This folder is excluded by `.gitignore`;
never commit or share the photos or the trained model in `outputs/demo/`.

Full instructions: [`docs/custom_demo.md`](../docs/custom_demo.md).
