# CycleGAN for Unpaired Image-to-Image Translation

A PyTorch course project exploring **horse-to-zebra and zebra-to-horse image translation without paired training examples**.

Developed by **Muhammad Haseeb Khan** as the final project for **DSCI-6602: Deep Learning and Artificial Intelligence**, taught by **Dr. Alex Bihlo** at Memorial University of Newfoundland.

**[Explore the notebook](DL_Project.ipynb)** · **[Original project summary](README.docx)**

## Project overview

CycleGAN learns mappings between two image domains while encouraging a translated image to reconstruct its original when mapped back. This project implements the core architecture and training objectives from [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593).

The notebook brings together image preprocessing, custom data loading, neural-network construction, alternating generator/discriminator optimization, checkpoint saving and visual inspection of generated images.

## Implementation

- **Two ResNet generators:** one for each translation direction, each with nine residual blocks, reflection padding, instance normalization and a final `Tanh` layer.
- **Two 70 × 70 PatchGAN discriminators:** assess local image patches in the horse and zebra domains.
- **Least-squares adversarial loss:** implemented with mean squared error.
- **Cycle-consistency loss:** L1 reconstruction loss for both translation cycles.
- **Identity loss:** encourages inputs already in the target domain to retain their appearance.
- **Training monitoring:** printed generator/discriminator losses and side-by-side input/A-to-B output images after each epoch.

### Configuration in the notebook

| Setting | Value |
|---|---|
| Image size | 256 × 256 RGB |
| Normalization | Pixel values scaled to [-1, 1] |
| Training epochs | 50 |
| Batch size | 1 |
| Optimizer | Adam |
| Learning rate | 0.0002 |
| Adam betas | (0.5, 0.999) |
| Cycle-consistency weight | 10 |
| Identity-loss coefficient | 5 per direction |
| Generator checkpoints | Saved every 10 epochs |
| Device selection | CUDA when available; otherwise CPU |

The learning-rate scheduler is configured to begin decay after epoch 50, so it does not reduce the learning rate during the configured 50-epoch run.

## Dataset

The notebook identifies the [Horse2Zebra dataset on Kaggle](https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset) as its source. Domain A contains horse images; domain B contains zebra images.

The committed dataset folders contain:

| Split | Images |
|---|---:|
| `archive/trainA/` | 1,067 |
| `archive/trainB/` | 1,334 |
| `archive/testA/` | 120 |
| `archive/testB/` | 140 |

Training uses the larger domain's length and cycles through each domain's sorted files by index. The data loader shuffles those indices; the implementation does not independently resample the second-domain image on every access.

## Repository contents

```text
.
├── DL_Project.ipynb      # Project explanation, implementation and saved training outputs
├── README.md            # Repository overview and usage notes
├── README.docx          # Original Word-format project summary
└── archive/
    ├── metadata.csv
    ├── trainA/          # Horse training images
    ├── trainB/          # Zebra training images
    ├── testA/           # Horse test images
    └── testB/           # Zebra test images
```

## Running the project

The notebook was written for **Google Colab**, with dataset and checkpoint paths under `/content/`. Its metadata records a GPU runtime. A GPU is recommended for training.

### Colab workflow

1. Open `DL_Project.ipynb` in Google Colab and select a GPU runtime.
2. Make the dataset ZIP available at `/content/archive.zip`, as expected by the extraction cell.
3. Check that extraction produces `/content/dataset/trainA/` and `/content/dataset/trainB/`. If the ZIP includes an extra folder level, adjust `dataset_root` accordingly.
4. Run the import, extraction and training cells in order.
5. Download the generator checkpoints written to `/content/` before the runtime ends.

### Local Jupyter workflow

Clone the repository and install the packages imported by the notebook:

```bash
git clone https://github.com/haseebkn/CycleGAN-for-Unpaired-Image-to-Image-Translation.git
cd CycleGAN-for-Unpaired-Image-to-Image-Translation
python -m pip install torch torchvision pillow matplotlib notebook
jupyter notebook DL_Project.ipynb
```

Use a compatible PyTorch/torchvision installation for your hardware. The original environment is not version-pinned.

For the included dataset, **skip the Colab ZIP-extraction cell** and change `dataset_root` to the local `archive` directory. Also change both checkpoint paths from `/content/` to an existing writable local directory before training. Check that both training folders contain images.

## Saved results and evaluation scope

The committed notebook contains saved training logs for **epochs 1–50**, checkpoint-save messages, a training-completion message and **50 image-output figures**. Its displayed figures compare a training input from domain A with its generated A-to-B translation.

These outputs support a qualitative review of the course experiment. The current notebook does **not**:

- Run a separate held-out test evaluation, despite including test-image folders.
- Report FID, LPIPS or other quantitative image-quality benchmarks.
- Include the trained checkpoint files in the repository.
- Provide a fixed-seed, version-locked reproduction of the original training run.

The notebook trains both directions, but its per-epoch visualization displays A-to-B examples only. Saved losses and selected images do not establish generalization quality or equivalence to the original paper's results.

## Reference

Jun-Yan Zhu, Taesung Park, Phillip Isola and Alexei A. Efros. **Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks**, ICCV 2017. [Paper](https://arxiv.org/abs/1703.10593).

This repository is an educational implementation of the CycleGAN approach.
