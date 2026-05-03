from setuptools import setup, find_packages

setup(
    name="pointnet2-kfs",
    version="0.1.0",
    description="PointNet++ for 3D point cloud segmentation (ShapeNetPart & KFS)",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20",
        "matplotlib>=3.5",
        "h5py>=3.0",
    ],
)
