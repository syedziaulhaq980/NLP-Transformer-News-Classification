# AG News Classification with DistilBERT & FastAPI

A production-oriented **NLP text classification project** that classifies news articles into four categories using a fine-tuned **DistilBERT Transformer** model.

The project compares a traditional **TF-IDF + Logistic Regression** baseline with a fine-tuned Transformer model and exposes the final model through a **FastAPI REST API**.

---

## 🚀 Project Overview

News articles are classified into four categories:

* 🌍 World
* ⚽ Sports
* 💼 Business
* 💻 Sci/Tech

The project follows an end-to-end machine learning workflow:

```text
News Text
    ↓
NLP Preprocessing
    ↓
TF-IDF + Logistic Regression
    ↓
Baseline Evaluation
    ↓
DistilBERT Tokenization
    ↓
Transformer Fine-Tuning
    ↓
Model Evaluation
    ↓
Hugging Face Model Repository
    ↓
FastAPI REST API
    ↓
Prediction + Confidence
```

---

## 🎯 Objectives

* Build a traditional NLP classification baseline.
* Understand and implement Transformer-based text classification.
* Fine-tune a pretrained DistilBERT model.
* Compare Transformer performance against a TF-IDF baseline.
* Save and reload the trained model.
* Host the trained model on Hugging Face.
* Build a REST API using FastAPI.
* Perform real-time news classification through an API endpoint.

---

## 📊 Dataset

The project uses the **AG News** dataset.

| Split    | Samples |
| -------- | ------: |
| Training | 120,000 |
| Test     |   7,600 |
| Total    | 127,600 |

The dataset contains four balanced classes:

| Label | Category |
| ----: | -------- |
|     0 | World    |
|     1 | Sports   |
|     2 | Business |
|     3 | Sci/Tech |

Dataset source:

**Hugging Face:** `fancyzhx/ag_news`

---

## 🧹 NLP Preprocessing

Traditional NLP preprocessing was performed for the baseline model.

The preprocessing pipeline included:

* Lowercasing
* Punctuation handling
* Text normalization
* Whitespace normalization
* Stop-word removal for TF-IDF

The Transformer model uses the original text with **DistilBERT's pretrained tokenizer**, rather than applying aggressive traditional preprocessing.

---

## 📈 Baseline Model — TF-IDF + Logistic Regression

A traditional NLP baseline was created using:

```text
TF-IDF Vectorization
        ↓
Logistic Regression
        ↓
4-Class Classification
```

TF-IDF feature matrix:

```text
Training: (120000, 64675)
Testing:  (7600, 64675)
```

### Baseline Accuracy

**91.57%**

| Class    | Precision | Recall | F1-score |
| -------- | --------: | -----: | -------: |
| World    |      0.93 |   0.90 |     0.92 |
| Sports   |      0.95 |   0.98 |     0.97 |
| Business |      0.89 |   0.88 |     0.88 |
| Sci/Tech |      0.89 |   0.90 |     0.89 |

---

## 🤖 Transformer Model — DistilBERT

The final classifier uses:

**`distilbert-base-uncased`**

The pretrained DistilBERT model was fine-tuned for the four AG News classes.

### Configuration

```text
Maximum sequence length: 128
Batch size: 32
Learning rate: 2e-5
Epochs: 2
Optimizer: AdamW
Loss: CrossEntropyLoss
```

### Transformer Pipeline

```text
Input News Article
        ↓
DistilBERT Tokenizer
        ↓
Input IDs + Attention Mask
        ↓
DistilBERT Transformer Encoder
        ↓
Classification Head
        ↓
4 News Categories
```

---

## 📊 Model Performance

### Test Accuracy

**94.50%**

The fine-tuned DistilBERT model improved the test accuracy over the TF-IDF baseline.

| Model                        |   Accuracy |
| ---------------------------- | ---------: |
| TF-IDF + Logistic Regression |     91.57% |
| Fine-tuned DistilBERT        | **94.50%** |

### DistilBERT Classification Report

| Class    | Precision | Recall | F1-score |
| -------- | --------: | -----: | -------: |
| World    |      0.97 |   0.94 |     0.96 |
| Sports   |      0.99 |   0.99 |     0.99 |
| Business |      0.92 |   0.91 |     0.91 |
| Sci/Tech |      0.90 |   0.94 |     0.92 |

The main classification confusion occurred between **Business** and **Sci/Tech**, where some news articles contain overlapping terminology and topics.

---

## 🤗 Model Hosting

The fine-tuned DistilBERT model and tokenizer are hosted on Hugging Face:

**Model:** [Ziaul1234/distilbert-agnews-classifier](https://huggingface.co/Ziaul1234/distilbert-agnews-classifier)

The FastAPI application downloads the model from Hugging Face when the application starts.
## ⚡ FastAPI

The trained Transformer model is exposed through a REST API.

### API Endpoints

| Method | Endpoint   | Description                       |
| ------ | ---------- | --------------------------------- |
| GET    | `/`        | API health/message                |
| POST   | `/predict` | Classify a news article           |
| GET    | `/docs`    | Interactive Swagger documentation |

---

## 📝 Prediction Example

### Request

```json
{
  "text": "Manchester United defeated Liverpool 3-1 in the Premier League match on Sunday."
}
```

### Response

```json
{
  "prediction": "Sports",
  "confidence": 0.9852118492126465
}
```

The confidence value represents the model's softmax probability for the predicted class.

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Hub
* DistilBERT
* Scikit-learn
* Pandas
* NumPy
* FastAPI
* Uvicorn
* Pydantic
* Jupyter / Google Colab
* Git & GitHub

---

## 📁 Project Structure

```text
NLP-Transformer-News-Classification/
│
├── app/
│   └── main.py
│
├── model/
│   └── distilbert_agnews_model/
│       └── Local model files
│
├── venv/
│
├── .gitignore
├── README.md
└── requirements.txt
```

> The trained model files and virtual environment are excluded from GitHub using `.gitignore`. The production API loads the model from Hugging Face.

---

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/syedziaulhaq980/NLP-Transformer-News-Classification.git
```

```bash
cd NLP-Transformer-News-Classification
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows:

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start FastAPI

```bash
uvicorn app.main:app --reload
```

### 6. Open Swagger

Open:

```text
http://127.0.0.1:8000/docs
```

You can then test the `/predict` endpoint interactively.

---

## 🔬 Example API Workflow

```text
Client
  │
  │ POST /predict
  │
  ▼
FastAPI
  │
  ▼
DistilBERT Tokenizer
  │
  ▼
Fine-Tuned DistilBERT
  │
  ▼
Softmax Probabilities
  │
  ▼
Predicted News Category
  │
  ▼
JSON Response
```

---

## 📌 Key Learning Outcomes

Through this project, I worked with:

* Traditional NLP preprocessing
* TF-IDF feature extraction
* Logistic Regression classification
* Transformer tokenization
* Attention masks
* DistilBERT architecture
* Transfer learning
* Transformer fine-tuning
* PyTorch training loops
* Model evaluation
* Confusion matrix analysis
* Softmax confidence scores
* Hugging Face model hosting
* FastAPI REST APIs
* API testing with Swagger
* Git and GitHub project management

---

## 🚀 Future Improvements

Potential future improvements include:

* Hyperparameter tuning
* Learning-rate scheduling
* Data augmentation
* Error analysis on misclassified articles
* Docker containerization
* Cloud deployment
* API monitoring
* Automated CI/CD

---

## 👨‍💻 Author

**Syed Ziaul Haq**

AI/ML | Data Science | NLP | Transformers | Generative AI

GitHub: `https://github.com/syedziaulhaq980`

---

## ⭐ Project Summary

This project demonstrates the complete journey from a traditional NLP baseline to a **fine-tuned Transformer model and production-style REST API**.

The final DistilBERT model achieved **94.50% test accuracy**, outperforming the TF-IDF + Logistic Regression baseline of **91.57%**.
