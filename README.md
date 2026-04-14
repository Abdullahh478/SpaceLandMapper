# SpaceLandMapper

SpaceLandMapper is an AI solution for land-use classification using EuroSAT imagery derived from Sentinel-2 satellite data. The project focuses on improving land classification accuracy through AI and presents the final solution in an interactive Streamlit dashboard.

## Project Overview

The prototype addresses the challenge of land-use mapping for areas such as:
- urban planning
- environmental conservation
- resource management

The system compares a baseline model with a Convolutional Neural Network (CNN), then uses the stronger CNN model as the final deployed solution in the dashboard.

## Features

- EuroSAT-based land classification prototype
- Baseline vs CNN model comparison
- Accuracy, Macro F1, and confusion matrix evaluation
- Live image-based prediction using the trained CNN
- Interactive Streamlit dashboard
- Prediction examples and model interpretation

## Project Structure

- `SRC/` - training, configuration, and data processing scripts
- `Dashboard/` - Streamlit dashboard application
- `Outputs/` - saved metrics, predictions, confusion matrices, and trained CNN model
- `Data/` - dataset split files and related data
- `README.md` - project overview and run instructions
- `requirements.txt` - project dependencies

## Final Model

The CNN was selected as the final model because it achieved better performance than the baseline model.

- Baseline Accuracy: 0.4085
- Baseline Macro F1: 0.3989
- CNN Accuracy: 0.8070
- CNN Macro F1: 0.7975

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt

```

### Start the Streamlit dashboard

```bash
streamlit run Dashboard/app.py
```

### Optional: Retrain the CNN model

Run this only if you want to regenerate the trained model and output files.

```bash
python SRC/train_cnn.py
```
## Dashboard Functionality

The dashboard includes:
- project overview
- model comparison
- confusion matrices
- live CNN prediction demo
- prediction examples
- land classes, limitations, and conclusion

## Limitations

The current prototype focuses on land classification using EuroSAT benchmark imagery. It does not yet implement full real-time land-use change monitoring.

## Future Work

Possible future improvements include:
- training for more epochs
- data augmentation
- broader Sentinel-2 testing
- land-use change detection over time
- deployment improvements

## Authors

SpaceLandMapper project team.
Abdullah Hassan
Altamash Khan
Hamzah Kounane
Mishal Zahra