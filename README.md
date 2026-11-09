# Printed Document Type Classification for Department Office Automation

## 1. Project Overview

This project presents a deep learning-based system for automatically classifying printed document images into six different categories: Email, Form, Letter, Memo, Note, and Report.

The system uses MobileNetV3-Small, a lightweight Convolutional Neural Network (CNN), trained using transfer learning on a filtered subset of the Tobacco3482 document dataset.

A web-based application developed using Streamlit allows users to upload document images and obtain the predicted document category along with its confidence score and class-wise probability distribution.

The primary objective is to simplify document categorization and support automation in departmental office environments.

## 2. Team and Institution

**Department:** Electronics and Computer Science  
**Institution:** Ramdeobaba University

## 3. Problem Statement

Manual sorting and categorization of printed documents in departmental offices is time-consuming and requires human effort. Handling large volumes of documents can also lead to inefficient organization and delays in document retrieval.

This project aims to develop an automated document classification system using deep learning to identify document categories from their visual appearance.

## 4. Objectives

- Develop an automated printed document classification system.
- Use deep learning and transfer learning for document categorization.
- Classify documents into six predefined categories.
- Evaluate model performance using standard classification metrics.
- Develop a user-friendly web application for document upload and prediction.
- Deploy the application online for easy accessibility.

## 5. Dataset

The project uses a filtered six-class subset of the Tobacco3482 document dataset.

**Dataset Link:**  
https://www.kaggle.com/datasets/d237prathmeshrane/document-classification-data

### Selected Document Categories

1. Email
2. Form
3. Letter
4. Memo
5. Note
6. Report

The original Tobacco3482 dataset contains 10 document categories. Six categories were selected for this project.

### Dataset Statistics

| Dataset Split | Number of Images |
|---|---:|
| Training | 2,144 |
| Testing | 539 |
| Total | 2,683 |

### Class-wise Distribution

| Class | Training Images | Testing Images |
|---|---:|---:|
| Email | 479 | 120 |
| Form | 344 | 87 |
| Letter | 453 | 114 |
| Memo | 496 | 124 |
| Note | 160 | 41 |
| Report | 212 | 53 |
| **Total** | **2,144** | **539** |

### Dataset Preprocessing

- Document images were resized to 224 × 224 pixels.
- Images were normalized before being passed to the model.
- Training and testing datasets were maintained separately.
- No overlap was present between training and testing images.

## 6. Methodology

The project follows a deep learning-based image classification pipeline.

### Step 1: Dataset Preparation

The Tobacco3482 dataset was filtered to retain six document categories. The images were organized into training and testing directories.

### Step 2: Image Preprocessing

Images were resized to 224 × 224 pixels and normalized to ensure consistent input dimensions and suitable numerical ranges.

### Step 3: Model Development

MobileNetV3-Small was selected as the classification architecture because of its lightweight design and computational efficiency.

Transfer learning was used to adapt the pretrained network to the six document categories.

### Step 4: Model Training

The model was trained using PyTorch. The classification layer was adapted to produce six output classes.

### Step 5: Model Evaluation

The trained model was evaluated on the held-out test dataset using accuracy, precision, recall, F1-score and confusion matrix analysis.

### Step 6: Application Development

A Streamlit application was developed to allow users to upload document images and obtain classification predictions.

### Step 7: Deployment

The application was deployed using Streamlit Community Cloud and the source code was maintained in GitHub.

## 7. Model Architecture

**Model Used:** MobileNetV3-Small

MobileNetV3-Small is a lightweight convolutional neural network designed for efficient image classification.

Important architectural components include:

- Depthwise Separable Convolutions
- Inverted Residual Blocks
- Squeeze-and-Excitation Modules
- Hard-Swish Activation Function
- Global Average Pooling
- Fully Connected Classification Layer

The final classification layer produces six output scores corresponding to the six document categories.

### Model Configuration

| Parameter | Configuration |
|---|---|
| Architecture | MobileNetV3-Small |
| Framework | PyTorch |
| Input Size | 224 × 224 |
| Number of Classes | 6 |
| Learning Approach | Transfer Learning |
| Output | Document Category |

## 8. Experimental Results

The final trained model was evaluated using 539 test images.

### Evaluation Metrics

| Metric | Result |
|---|---:|
| Test Accuracy | 91.09% |
| Macro F1-score | 90.00% |
| Weighted F1-score | 90.96% |
| Training Accuracy | 94.59% |

### Per-Class F1-score

| Document Class | F1-score |
|---|---:|
| Email | 96.72% |
| Form | 94.32% |
| Letter | 86.11% |
| Memo | 91.19% |
| Note | 83.78% |
| Report | 87.85% |

### Evaluation Parameters

- **Accuracy:** Measures the proportion of correctly classified documents.
- **Precision:** Measures the reliability of predictions for each class.
- **Recall:** Measures the proportion of actual class samples correctly identified.
- **F1-score:** Represents the harmonic mean of precision and recall.
- **Confusion Matrix:** Identifies correct classifications and misclassifications between document categories.

Macro and weighted F1-scores were considered to evaluate performance across classes with different sample counts.

## 9. Web Application

The web application was developed using Streamlit.

### Application Features

- Upload document images.
- Capture document images using a camera.
- Display the uploaded document preview.
- Predict the document category.
- Display the predicted class and confidence score.
- Display class-wise probability distribution.

### Supported Document Categories

Email, Form, Letter, Memo, Note and Report.

### Live Application

https://document-type-classifier.streamlit.app/

## 10. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| Torchvision | Model architecture and image transformations |
| MobileNetV3-Small | Document classification |
| Streamlit | Web application development |
| Pillow (PIL) | Image processing |
| NumPy | Numerical operations |
| Kaggle | Dataset and model development environment |
| GitHub | Source code management |
| Streamlit Community Cloud | Application deployment |

## 11. Repository Structure

```text
document-classifier/
│
├── app.py
├── document_classifier_final.pth
├── requirements.txt
├── README.md
│
├── homepage.png
├── Document upload.png
└── prediction results and class probabilities.png
```

## 12. Installation and Execution

### Prerequisites

- Python installed on the system.
- Git installed on the system.
- Required project files downloaded from the repository.

### Step 1: Clone the Repository

```bash
git clone https://github.com/PrathmeshRane-xyz/document-classifier.git
```

### Step 2: Navigate to the Project Directory

```bash
cd document-classifier
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

The application will start locally and can be accessed through the URL displayed in the terminal, usually:

http://localhost:8501

**Note:** The trained model weights file, `document_classifier_final.pth`, must be present in the project directory when running the application locally.

## 13. Challenges and Limitations

### Challenges

- Document categories can have similar visual layouts.
- Some document classes contain fewer training samples.
- Variations in scanned images and document formatting can affect classification.
- Selecting a lightweight architecture while maintaining good classification performance required experimentation.

### Limitations

- The system supports only six predefined document categories.
- The model performs image-based classification and does not extract document text.
- Classification performance may vary for poor-quality images and document types not included in the training dataset.

## 14. Future Scope

- Extend the model to support all ten original Tobacco3482 categories and additional document types.
- Improve performance on visually similar document categories.
- Integrate Optical Character Recognition (OCR) for extracting text from documents.
- Explore deployment on edge devices for efficient local processing.
- Integrate the classifier into departmental document management systems.
- Explore automated document organization and retrieval.

## 15. Conclusion

This project demonstrates the application of deep learning for automated printed document classification.

A MobileNetV3-Small-based model was developed using transfer learning and trained on a filtered subset of the Tobacco3482 dataset containing six document categories.

The final evaluation achieved 91.09% test accuracy, with a macro F1-score of 90.00% and a weighted F1-score of 90.96%.

A Streamlit-based web application was also developed and deployed online, providing an accessible interface for document upload and classification.

The proposed system demonstrates the potential of lightweight deep learning models for automating document categorization in departmental office environments.

## 16. Acknowledgement

We express our gratitude to Ramdeobaba University and the Department of Electronics and Computer Science for providing the academic environment and support required to carry out this project.

## References

1. S. J. A. R. S. Kumar et al., "Tobacco3482: A Dataset for Document Image Classification." The original dataset and its associated publication should be cited using the verified bibliographic details of the source.

2. A. Howard et al., "Searching for MobileNetV3," Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019.

3. PyTorch Documentation. https://pytorch.org/docs/

4. Streamlit Documentation. https://docs.streamlit.io/
