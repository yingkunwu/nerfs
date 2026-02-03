# NeRFs

This repository is a study of **NeRF, DS-NeRF, NeRF++, NeRF in the Wild (NeRFW), and NSFF**. It integrates implementations from various state-of-the-art resources into a single framework. Many parts of the code are borrowed from:

* [nerf_pl](https://github.com/kwea123/nerf_pl) & [nsff_pl](https://github.com/kwea123/nsff_pl)
* [DSNeRF](https://github.com/dunbar12138/DSNeRF)
* [nerfplusplus](https://github.com/Kai-46/nerfplusplus)
* [Neural-Scene-Flow-Fields](https://github.com/zhengqili/Neural-Scene-Flow-Fields)

> 💡 **Feature:** Different versions of NeRF can be executed simply by switching the configuration file.

---

## Environment Setup

A Docker environment is provided to ensure consistency across different NeRF implementations.

1. **Build the image:** `docker build -t colmap docker/`
2. **Launch the container:** `./docker/run_container.sh`

## Data Preparation

For convenience, I have only tested a specific dataset for each NeRF version, though they are designed to be used interchangeably.

### NeRF & DS-NeRF
* **Dataset:** Download LLFF dataset from [here](https://github.com/Fyusion/LLFF).
* **Depth Assisted Loss:** To use DS-NeRF's depth loss, we need to generate depth maps from images:
    1. Follow the [Preparation Guide](preprocess/README.md).
    2. Run: `python preprocess/process_llff.py --root_dir <path/to/ROOT_DIR>`

### NeRF++
* **Dataset:** Download from [Tanks and Temples](https://www.tanksandtemples.org).
* **Preprocessing:** 1. Follow the [Preparation Guide](preprocess/README.md).
    2. Run: `python preprocess/normalize_cam_dict.py --root_dir <path/to/ROOT_DIR>`
    * *Note: This normalizes poses into a unit circle as proposed by the original paper. Results are saved in `split_data` under the root directory.*

### NeRF in the Wild (NeRFW)
* **Dataset:** Download PhotoTourism dataset from [UBC PhotoTourism](https://www.cs.ubc.ca/~kmyi/imw2020/data.html).

### NSFF (Neural Scene Flow Fields)
* **Dataset:** Currently using the "Kid" series from [nsff_pl v2.0](https://github.com/kwea123/nsff_pl/releases/tag/v2.0).

---

## Training & Inference

### Training
Use the YAML files in `configs/` to specify your model and dataset preferences.
```bash
# Example: Training DS-NeRF on LLFF
python train.py --config configs/train_dsnerf.yaml
```

### Inference
Weights and logs are stored in the logs/ directory.
```bash
# Example: Run inference on a specific version
python inference.py --log_path logs/dsnerf/version_<ID>
```

## Demo

* NeRF
    <img src="demo/nerf.gif" alt="nerf" width="800"/>
* NeRF++
    <img src="demo/nerf++.gif" alt="nerf++" width="800"/>
* NeRFW
    <img src="demo/nerfw.gif" alt="nerfw" width="800"/>
* NSFF
    <img src="demo/nsff.gif" alt="nsff" width="800"/>
    * **Note on NSFF**: My implementation of NSFF currently has performance issues. I am very welcome to feedback or suggestions on how to improve this section!