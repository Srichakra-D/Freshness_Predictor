# Fruit Freshness Predictor

This repository contains a freshness classifier for fruit and vegetable images.
The app loads a saved PyTorch model from `model.pth`, accepts an uploaded image,
and predicts:

- the fruit or vegetable class
- whether the item is fresh or spoiled

The main runnable app is implemented with Streamlit in `main.py`.

## Repository Contents

| File | Purpose |
| --- | --- |
| `main.py` | Streamlit inference app for uploading an image and getting a prediction. |
| `model.pth` | Saved PyTorch model state dictionary used by the app. |
| `project-compvis-lec.ipynb` | Kaggle-style training and evaluation notebook for the fresh/spoiled classifier. |
| `freshness_predictor.ipynb` | Earlier notebook version of the predictor with a Tkinter interface. |
| `requirements.txt` | Base Python dependencies currently tracked in the repo. |

## Model Summary

The deployed model uses a VGG16 feature extractor with two custom output heads:

- fruit class head: 9 classes
- freshness head: 2 classes

`model.pth` was checked as a PyTorch `state_dict` with compatible output shapes:

- `block2.3.weight`: `(9, 128)`
- `block3.3.weight`: `(2, 32)`

## Setup

Create and activate a Python 3.10+ virtual environment, then install
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

The Streamlit app expects `model.pth` to be beside `main.py`, which is the
repository root in the current layout.

## Run the App

Start the Streamlit app from the repository root:

```bash
streamlit run main.py
```

Then open the local URL shown by Streamlit and upload a `jpg`, `jpeg`, or `png`
image.

## Training Notebook Notes

`project-compvis-lec.ipynb` is a Kaggle notebook built around the
`fresh-and-stale-classification` dataset. It contains:

- data loading from Kaggle paths
- train, validation, and test splits
- image augmentation and PyTorch `DataLoader` setup
- VGG16-based model training
- loss and accuracy plots
- confusion matrices
- classification reports

The notebook metadata indicates a Kaggle GPU environment and references dataset
ID `3371317`. It should be run in Kaggle or in a local environment where the same
dataset paths are available.

## Label Order

The training notebook derives fruit labels with `LabelEncoder`. The Streamlit
app uses the same class order:

```text
apples, banana, bittergroud, capsicum, cucumber, okra, oranges, potato, tomato
```

Keep this order unchanged unless the model is retrained with a different
encoder.

## Current Caveats

- The app preprocessing follows the inference transform from the training
  notebook: resize to `224x224`, convert to tensor, and identity normalization.
- `project-compvis-lec.ipynb` contains saved notebook outputs and plots, making
  it larger than a clean source-only notebook.
- The final saved notebook report shows high freshness accuracy, but fruit-class
  test accuracy is low for several classes. The saved test labels include naming
  inconsistencies such as `patato` and `tamto`, so verify label consistency
  before reporting model performance.
