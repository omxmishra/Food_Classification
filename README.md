# 🍽️ Food-101 Image Classifier

A transfer-learning image classifier that identifies 101 food categories from a single photo, built with MobileNetV2, fine-tuned on the Food-101 dataset, and served through an interactive Streamlit app with Grad-CAM visual explanations.

## Overview

This project takes a pretrained MobileNetV2 backbone and adapts it to the Food-101 dataset (101,000 images across 101 food classes) using a two-stage transfer learning approach: first training a new classification head on top of a frozen backbone, then fine-tuning the top layers of the backbone at a lower learning rate. The final model is wrapped in a Streamlit interface that returns top-K predictions with confidence scores and a Grad-CAM heatmap showing which regions of the image influenced the prediction.

## Features

- MobileNetV2 backbone pretrained on ImageNet, adapted via transfer learning
- Two-stage training: frozen-backbone warmup followed by selective fine-tuning
- Data augmentation pipeline (flip, rotation, zoom, contrast, translation)
- Stratified train/validation/test split to preserve class balance
- Full evaluation suite: accuracy, top-5 accuracy, precision, recall, F1, confusion matrix, most-confused class pairs, and misclassified sample inspection
- Grad-CAM implementation for visual model interpretability
- Interactive Streamlit app with top-K predictions and heatmap overlay
- GPU-accelerated training via WSL2 + CUDA

## Results

| Metric | Value |
|---|---|
| Validation Accuracy (fine-tuned) | 63.21% |
| Validation Top-5 Accuracy (fine-tuned) | 86.13% |
| Test Accuracy | see `models/food101_test_metrics.csv` |
| Test Top-5 Accuracy | see `models/food101_test_metrics.csv` |
| Test Precision (weighted) | see `models/food101_test_metrics.csv` |
| Test Recall (weighted) | see `models/food101_test_metrics.csv` |
| Test F1 (weighted) | see `models/food101_test_metrics.csv` |

Food-101 is a genuinely difficult dataset with 101 visually similar classes and noisy training labels by design. A MobileNetV2 transfer-learning baseline in the low-to-mid 60s is a reasonable, honest result for this class of lightweight backbone; higher published numbers typically use heavier architectures (ResNet50, EfficientNet) with longer fine-tuning schedules.

## Project Structure

```
Food-101-Classification/
├── app/
│   └── app.py              Streamlit inference app
├── data/                   Food-101 image data (class-per-folder)
├── models/                 Saved checkpoints, final model, metrics
├── notebooks/
│   └── Food101_Classification.ipynb   Original exploratory notebook
├── src/
│   ├── config.py            Paths and hyperparameters
│   ├── pipeline.py          Data loading, dataset pipeline, model, training, evaluation
│   └── gradcam.py           Grad-CAM generation and heatmap overlay
├── requirements.txt
└── .gitignore
```

## Dataset

[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) consists of 101 food categories with 1,000 images each. Images are expected in `data/<class_name>/<image files>` format, one folder per class.

## Setup

```bash
git clone https://github.com/omxmishra/Food-101-Classification.git
cd Food-101-Classification

python3 -m venv food101_env
source food101_env/bin/activate

pip install -r requirements.txt
```

GPU acceleration on Windows requires running inside WSL2, since native Windows TensorFlow builds (2.11+) no longer support CUDA directly.

## Training

Update `DATA_DIR` in `src/config.py` to point to your local dataset location, then run from the project root:

```bash
python -m src.pipeline
```

This runs the full pipeline: dataframe construction, stratified split, `tf.data` pipeline with augmentation, frozen-backbone training, fine-tuning, evaluation, and saves the following into `models/`:

- `mobilenet_best.keras` — best checkpoint from the frozen-backbone stage
- `mobilenet_finetuned.keras` — best checkpoint from the fine-tuning stage
- `food101_mobilenetv2_finetuned.keras` — final saved model
- `class_names.json` — ordered class label mapping
- `food101_test_metrics.csv` — final test-set metrics

## Running the App

```bash
streamlit run app/app.py
```

Upload a food image to get top-K predictions with confidence scores and an optional Grad-CAM heatmap overlay showing the regions the model focused on.

## Tech Stack

- TensorFlow / Keras
- MobileNetV2 (ImageNet weights)
- Streamlit
- OpenCV, Pillow
- scikit-learn
- pandas, NumPy

## Grad-CAM

Model interpretability is handled via Grad-CAM, which highlights the spatial regions in the input image that most strongly influenced the predicted class. This is computed by taking gradients of the top predicted class with respect to the last convolutional layer's activations, then weighting and overlaying them as a heatmap on the original image.

## Future Improvements

- Swap MobileNetV2 for a heavier backbone (EfficientNetB0/B3) to close the accuracy gap with published benchmarks
- Add test-time augmentation for a small accuracy boost at inference
- Add per-class precision/recall breakdown to the Streamlit app
- Containerize the app with Docker for easier deployment
- Add a FastAPI inference endpoint alongside the Streamlit UI

## Author

**Om Mishra**
[GitHub](https://github.com/omxmishra) · [LinkedIn](https://linkedin.com/in/om--mishra) · [X](https://x.com/BuildWithOm)
