# Fruit Freshness Predictor

A Streamlit application that uses a multitask PyTorch model to identify produce
and classify it as fresh or spoiled.

## Features

- Upload `jpg`, `jpeg`, or `png` images through a browser interface.
- Predict one of nine fruit and vegetable classes.
- Predict freshness as `Fresh` or `Spoiled`.
- Show softmax confidence for both model outputs.
- Warn when either prediction has less than 55% confidence.

## Architecture

The deployed model uses VGG16 convolutional features followed by a shared dense
layer and two classification heads:

| Head | Classes |
| --- | --- |
| Produce | apples, banana, bittergroud, capsicum, cucumber, okra, oranges, potato, tomato |
| Freshness | Fresh, Spoiled |

`inference.py` owns the model architecture, preprocessing, checkpoint loading,
and prediction result. `main.py` contains only the Streamlit interface.

The committed `model.pth` checkpoint contains a compatible PyTorch state
dictionary. Images are converted to RGB, resized to `224x224`, and converted to
tensors using the same identity normalization as the training notebook.

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

For development and testing:

```bash
pip install -r requirements-dev.txt
pytest
```

## Repository Contents

| Path | Purpose |
| --- | --- |
| `main.py` | Streamlit user interface |
| `inference.py` | Reusable model and inference pipeline |
| `model.pth` | Trained model state dictionary |
| `project-compvis-lec.ipynb` | Kaggle training and evaluation notebook |
| `freshness_predictor.ipynb` | Earlier Tkinter prototype |
| `tests/` | Automated inference and checkpoint tests |

## Training and Evaluation

The Kaggle notebook uses the
`fresh-and-stale-classification` dataset, dataset ID `3371317`. It now:

- normalizes the source label typos `patato` and `tamto`;
- fits one `LabelEncoder` and reuses it for train, validation, and test data;
- uses a training method that does not override `torch.nn.Module.train`;
- computes epoch accuracy from sample counts rather than cumulative metrics.

Saved notebook outputs were removed because the previous fruit accuracy was
computed with inconsistent train and test encoders. Retrain and rerun all cells
in Kaggle before publishing updated performance numbers.

## Limitations

- Confidence values are raw softmax scores and are not calibrated probabilities.
- The model always chooses a supported class; the warning is not an
  out-of-distribution detector.
- Performance depends on lighting, framing, and whether the image resembles the
  training dataset.
- The test dataset does not contain every training class, so per-class reports
  must explicitly include only labels represented by the evaluated split.
