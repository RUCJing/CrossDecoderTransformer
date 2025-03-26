# Overview

This project (CrossDecoderTransformer) is based on the research of "Online Diagnose Patient Satisfaction Prediction Based on Multimodal Data". The core model architecture is defined in `model.py` and trained using `model_train.py`. The trained models are saved in the `models` directory, facilitating subsequent performance evaluation on test data using `model_eval.py`.

# Project Structure

## Folders
- `dataset`: This directory is used to store the datasets required for the project(https://huggingface.co/datasets/FireflyLiu/C_Online_Diagnose/blob/main/README.md).
- `font`: Stores font-related files used in visualizations.
- `image`: Primarily stores image files generated during the model evaluation process, such as visualization images.
- `models`: Used to save the trained model files. In the project, the trained models are stored in this directory for subsequent model loading and evaluation.

## Files
- `model_eval.py`: A script for model evaluation. It loads the trained model, evaluates it, and plots the confusion matrix and ROC curve.
- `model_train.py`: A script for model training. It saves the trained model to the `models` directory.
- `model.py`: The specific implementation of the CrossDecoderTransformer model.
- `params.py`: Stores various parameter settings for the project, facilitating unified management and modification of parameters.
- `text_featuring.py`: Performs feature processing and generates a dataset suitable for PyTorch training.
- `preprocessing.py`: For Chinese text, it performs character segmentation and saves the correspondences between Chinese characters and tokens, as well as labels and numerical values. For English text, other customized tokenizers can be used.

# Training Process

First, execute the following command:
```bash
pip install -r requirements.txt
```

Before training on Chinese text, run:
```bash
python preprocessing.py
```

To train the model, use the following command:
```bash
python model_train.py
```

To evaluate the model, use the following command:
```bash
python model_eval.py
```