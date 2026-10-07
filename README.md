# PatchCore — Pretrained Models for MVTec AD

Pretrained PatchCore models for all **15 MVTec AD categories**, with inference code, evaluation, and anomaly-map visualization.

This repository adapts the original PatchCore implementation to provide a reproducible workflow for using saved models. No retraining is required.

## Download the Models

**[Download pretrained models from OneDrive (967 MB)](https://univr-my.sharepoint.com/:u:/g/personal/muhammad_aqeel_univr_it/IQCvi-G3A_B7TJq_QZmvWiiIAVDLLSmYwefUNGqJXCVM1h4?e=DU3l2r)**

Download `patchcore_mvtec_models.tar.gz`, move it into the cloned repository, and extract it:

```bash
tar -xzf patchcore_mvtec_models.tar.gz
```

This creates the `models/` folder containing all 15 category models.

Each category folder—for example, `models/mvtec_bottle/`—contains:

- `nnscorer_search_index.faiss`: the trained feature memory bank.
- `patchcore_params.pkl`: the model configuration.

Keep both files together.

The ImageNet-pretrained backbone weights are loaded separately. They must be available in the local Torch cache or downloadable on first use.

**These models use the included standalone PatchCore implementation. Compatibility with anomalib has not been tested.**

## Installation

Clone the repository:

```bash
git clone https://github.com/aqeeelmirza/patchcore.git
cd patchcore
```

Create an environment and install dependencies:

```bash
conda create -n patchcore-inference python=3.10 -y
conda activate patchcore-inference

python -m pip install torch==2.7.1 torchvision==0.22.1
python -m pip install -r requirements.txt timm
```

The models were verified using Python 3.10, PyTorch 2.7.1, torchvision 0.22.1, and CPU FAISS.

The examples below run in **Bash** and use **GPU 0 for feature extraction** and **CPU FAISS for nearest-neighbor search**. They do not require GPU FAISS.

## Prepare MVTec AD

Download the dataset from the [official MVTec AD website](https://www.mvtec.com/company/research/datasets/mvtec-ad).

Keep the original directory structure. For each category, the dataset should contain:

- `<category>/train/good/`
- `<category>/test/good/`
- `<category>/test/<defect_type>/`
- `<category>/ground_truth/<defect_type>/`

Set the dataset root in your terminal:

```bash
mvtec_root="/path/to/mvtec"
```

The root must contain the category folders, such as `bottle`, `cable`, and `capsule`.

The included evaluator requires the test images and ground-truth masks.

## Evaluate One Category

Run this command from the repository root:

```bash
PYTHONPATH="$PWD/src" python bin/load_and_evaluate_patchcore.py \
  --gpu 0 \
  outputs/bottle \
  patch_core_loader \
  -p models/mvtec_bottle \
  dataset \
  --resize 256 \
  --imagesize 224 \
  --num_workers 2 \
  -d bottle \
  mvtec "$mvtec_root"
```

Results are saved to:

```text
outputs/bottle/results.csv
```

To evaluate another category, change the model path, category name, and output directory.

**Use `--resize 256 --imagesize 224` for the supplied model set.**

## Generate Anomaly Visualizations

Add `--save_segmentation_images` to export visualizations:

```bash
MPLBACKEND=Agg PYTHONPATH="$PWD/src" \
python bin/load_and_evaluate_patchcore.py \
  --gpu 0 \
  --save_segmentation_images \
  outputs/bottle_visualizations \
  patch_core_loader \
  -p models/mvtec_bottle \
  dataset \
  --resize 256 \
  --imagesize 224 \
  --num_workers 2 \
  -d bottle \
  mvtec "$mvtec_root"
```

The exported figures contain three panels:

1. Preprocessed input image.
2. Ground-truth defect mask.
3. Predicted anomaly map.

The input is resized and center-cropped, so the predicted map corresponds to the cropped image.

### Using Maps for Vision-Language Q&A

For workflows using models such as SmolVLM, provide the input image and predicted anomaly map **without the ground-truth panel**.

The current exporter produces combined evaluation figures. It does not export separate heatmap-only or overlay images.

The evaluator normalizes anomaly maps across the evaluated test images. The plotting function also automatically scales each displayed map. Therefore:

- Color intensity represents relative anomaly scores.
- Colors are not calibrated defect probabilities.
- A heatmap alone does not provide a thresholded normal/anomalous decision.
- Colors in the exported figures should not be used to compare absolute anomaly severity across images.

## Evaluate All 15 Categories

Run the following in **Bash**, from the repository root:

```bash
mvtec_root="/path/to/mvtec"

classes=(
  bottle cable capsule carpet grid
  hazelnut leather metal_nut pill screw
  tile toothbrush transistor wood zipper
)

model_args=()
dataset_args=()

for cls in "${classes[@]}"; do
  model_args+=(-p "models/mvtec_$cls")
  dataset_args+=(-d "$cls")
done

PYTHONPATH="$PWD/src" python bin/load_and_evaluate_patchcore.py \
  --gpu 0 \
  outputs/all_classes \
  patch_core_loader "${model_args[@]}" \
  dataset \
  --resize 256 \
  --imagesize 224 \
  --num_workers 2 \
  "${dataset_args[@]}" \
  mvtec "$mvtec_root"
```

This loads the existing models and evaluates each category without retraining.

Results are saved to:

```text
outputs/all_classes/results.csv
```

## Verified Results

All 15 supplied models were successfully loaded and evaluated.

The following values are arithmetic means across categories:

| Metric | Mean AUROC |
|---|---:|
| Image-level anomaly detection | 99.17% |
| Pixel-level localization: all test images | 98.11% |
| Pixel-level localization: anomalous images only | 97.36% |

Per-category results are provided in [verified_results.csv](verified_results.csv).

Small numerical differences may occur across software and hardware environments.

## Model Configuration

| Parameter | Value |
|---|---|
| Backbone | ImageNet-pretrained Wide-ResNet50-2 |
| Extracted layers | `layer2`, `layer3` |
| Initial resize | 256 |
| Center-cropped input | 224 × 224 |
| Pretraining embedding dimension | 1024 |
| Target embedding dimension | 1024 |
| Patch size | 3 |
| Nearest neighbors | 1 |
| Saved memory-bank format | FAISS |

PatchCore uses a pretrained feature extractor and a category-specific memory bank of normal-image patch features.

## Supported Categories

`bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`,
`leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`,
`transistor`, `wood`, and `zipper`.

## Troubleshooting

### Missing FAISS

```bash
python -m pip install faiss-cpu
```

### Missing timm

```bash
python -m pip install timm
```

### PatchCore Cannot Be Imported

Run commands from the repository root with `PYTHONPATH` pointing to `src`. To check the import:

```bash
PYTHONPATH="$PWD/src" python -c "import patchcore; print(patchcore.__file__)"
```

### Missing Backbone Weights

The model loader reconstructs the pretrained backbone. Allow the initial weight download, or make the matching weights available in the Torch cache.

### torchvision Deprecation Warnings

The code uses the older `pretrained` argument. These warnings did not prevent loading or evaluation in the verified environment.

## Acknowledgments and License

This repository is based on the [original PatchCore implementation](https://github.com/amazon-science/patchcore-inspection) and the paper *Towards Total Recall in Industrial Anomaly Detection*.

PatchCore is the original authors’ method. This repository provides an adapted workflow and saved models evaluated on MVTec AD.

The upstream license and applicable notices must be retained when redistributing the code. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for terms and attribution.

MVTec AD is distributed separately under its own dataset terms.
