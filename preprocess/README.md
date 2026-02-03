# Dataset Preprocess

**This preprocessing is only required if your dataset does not have camera poses information or depth and optical flow are required.**
This dataset preprocessing are mainly adapted from [nsff_pl](https://github.com/kwea123/nsff_pl).

## Data preparation

Create a root directory (e.g. `foobar`), create a folder named `frames` and prepare your images (it is recommended to have at least 30 images) under it, so the structure looks like:

```bash
└── ROOT_DIR
    └── frames
        ├── 00000.png
        ├── ...
        └── 00029.png
```

The image names can be arbitrary, but the lexical order should be the same as time order! E.g. you can name the images as `a.png`, `c.png`, `dd.png` but the time order must be `a -> c -> dd`.
  
### Depth
  
The instructions and code are borrowed from [DPT](https://github.com/intel-isl/DPT).

Download the model weights from [here](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt) and put it in `preprocess/depth/weights/`.

### Optical Flow
The instructions and code are borrowed from [RAFT](https://github.com/princeton-vl/RAFT).

Download `raft-things.pth` from [google drive](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT) and put it in `preprocess/flow/`.

### Prediction
  
After preparing the images and the model weights, we can automate the whole process by a single command `python preprocess/run_preprocess.py --root_dir <path/to/ROOT_DIR>`.
**Make sure your images are under the frames folder. If not the code cannot find the images and will crash.**

Finally, your root directory will have all of this:

```bash
└── ROOT_DIR
    ├── frames (original images, not used)
    │   ├── 00000.png
    │   └── 00029.png
    ├── images_resized (resized images, not used)
    │   ├── 00000.png
    │   └── 00029.png
    ├── images_undistorted (the images to use in training)
    │   ├── images
    │   │   ├── 00000.png
    │   │   └── 00029.png
    │   └── sparse (the poses to use in training)
    │       ├── cameras.bin
    │       ├── images.bin
    │       ├── points3D.bin
    │       └── frames.bin
    ├── masks (not used but do not delete)
    │   ├── 00000.png.png
    │   └── 00029.png.png
    ├── database.db
    ├── sparse (not important)
    │   └── 0
    │       ├── cameras.bin
    │       ├── images.bin
    │       ├── points3D.bin
    │       └── project.ini
    ├── disps
    │   └── 00000.png
    ├── flow_fw (for nsff training)
    │   └── 00000.flo
    └── flow_bw (for nsff training)
        └── 00001.flo
```
