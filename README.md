# SAM2-Based Abdominal Wall Tumor Segmentation

This project applies the SAM2 model to the segmentation of abdominal wall tumors. The workflow consists of the following steps:

1. **Input**: Load abdominal tumor NIfTI (.nii.gz) images.
2. **Preprocessing**: Crop the images into PNG format.
3. **User Interaction**: Utilize a UI for selecting positive and background seeds.
4. **Inference**: Generate the final segmentation mask as a NIfTI (.nii.gz) output using SAM2 inference.

## Requirements

### Software Dependencies
- Python 3.10+
- PyTorch 2.0+
- SAM2 model
- NumPy
- ITK
- Matplotlib
- PyQt5 (for UI interaction)

### Installation
```bash
pip install torch torchvision torchaudio
pip install opencv-python numpy SimpleITK matplotlib PyQt5
```

## Hardware Configuration
- **GPU**: NVIDIA A4000 (or equivalent)
- **CUDA Version**: 12.6+
- **VRAM**: 24GB+ (recommended for large NIfTI image processing)
- **RAM**: 32GB+ (recommended for smooth UI and processing)

## Usage
To run the segmentation pipeline:
```bash
python interactiveUI_v3.py
