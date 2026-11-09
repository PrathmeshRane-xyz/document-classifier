
# Printed Document Type Classification for Department Office Automation

## Project Overview

This project develops a deep learning-based system that automatically classifies printed document images into six categories: Email, Form, Letter, Memo, Note, and Report.

The system uses a **MobileNetV3-Small** convolutional neural network trained using the Tobacco3482 document dataset. A Streamlit web application allows users to upload a document image or capture one using a camera and receive its predicted category along with the model's confidence score.

## Team Members

Department of Electronics and Computer Science  
Ramdeobaba University

- Prathamesh Rane
- Prashil Patil
- Ritesh Bande
- Pranav Bangare

## Problem Statement

Manual sorting and categorization of printed documents in departmental offices can be time-consuming. This project aims to automate document categorization using deep learning, making document organization faster and more efficient.

## Dataset

The project uses a filtered subset of the **Tobacco3482 dataset**, containing six document categories:

1. Email
2. Form
3. Letter
4. Memo
5. Note
6. Report

- Training images: 2,144
- Testing images: 539
- Total images: 2,683

The dataset was divided into training and testing sets. The images were resized to 224 × 224 pixels and normalized before being supplied to the model.

**Dataset access:** Obtain the Tobacco3482 dataset from its original research dataset source. This project uses a filtered six-class subset. The dataset is not included in this repository.

## Model Architecture

- Model: MobileNetV3-Small
- Framework: PyTorch
- Input image size: 224 × 224
- Number of output classes: 6
- Transfer learning was used for document image classification.

The trained model weights are provided in:

`document_classifier_final.pth`

## Experimental Results

The final model was evaluated on 539 test images.

| Evaluation Metric | Result |
|---|---:|
| Test Accuracy | 91.09% |
| Macro F1-score | 90.00% |
| Weighted F1-score | 90.96% |
| Training Accuracy | 94.59% |

## Application Features

- Upload document images in JPG or PNG format.
- Capture document images using a camera.
- Display the uploaded document preview.
- Predict one of six document categories.
- Display prediction confidence.
- Display probability scores for all six categories.

## Technologies Used

- Python
- PyTorch
- Torchvision
- Streamlit
- PIL (Pillow)
- NumPy

## Repository Structure

```text
document-classifier/
│
├── app.py
├── document_classifier_final.pth
├── requirements.txt
└── README.md
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/PrathmeshRane-xyz/document-classifier.git
cd document-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your web browser.

## Deployed Application

The project is also deployed using Streamlit Community Cloud.

**Live Application:**  
https://document-type-classifier.streamlit.app/

## Future Scope

- Expand the system to support additional document categories.
- Improve classification performance on visually similar document types.
- Explore deployment on edge devices.
- Integrate OCR for future document text extraction capabilities.

## Acknowledgement

We acknowledge Ramdeobaba University and the Department of Electronics and Computer Science for providing the academic environment and support to carry out this project.
