# PointNet++ for 3D Point Cloud Segmentation

Encoder-decoder PointNet++ with multi-scale grouping and attention for 3D point cloud part segmentation (ShapeNetPart & custom KFS dataset).

## Project Structure

```
project2/
├── config/
│   └── default.py            # Hyperparameters & paths (dataclass)
├── src/                      # Python package
│   ├── ops/                  # Core PointNet++ operators (SA, FP, attention)
│   ├── models/               # Segmentation & classification models
│   ├── data/                 # Dataset classes & transforms
│   └── utils/                # Metrics & visualization
├── scripts/                  # Entry points
│   ├── train.py              # Train on ShapeNetPart or KFS
│   ├── evaluate.py           # Evaluate on test set
│   ├── inference.py          # Run on a single point cloud
│   ├── export.py             # Export to TorchScript / ONNX
│   └── prepare_data.py       # PCD → npy conversion & labeling
├── tests/                    # Unit tests for core ops
├── checkpoints/              # Trained weights
├── data/                     # Datasets
│   ├── hdf5_data/            # ShapeNetPart HDF5 files
│   ├── raw/                  # Raw PCD files
│   └── processed/            # Processed .npy files
├── outputs/                  # Exported models (.ts, .onnx)
├── logs/                     # TensorBoard logs
├── config/                   # Configuration dataclasses
├── requirements.txt
└── setup.py                  # pip install -e .
```

## Quick Start

```bash
pip install -e .
python scripts/train.py
python scripts/evaluate.py
python scripts/export.py
```

## KFS Custom Dataset

```bash
# 1. Convert PCD → npy
python scripts/prepare_data.py --pcd_dir data/raw --out_dir data/processed

# 2. Interactive labeling
python scripts/prepare_data.py --label

# 3. Train on KFS
python scripts/train.py --kfs
```
