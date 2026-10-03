from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI(
    title="AG News Classification API",
    description="News classification using a fine-tuned DistilBERT model.",
    version="1.0.0"
)

MODEL_PATH = "Ziaul1234/distilbert-agnews-classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()


class NewsRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "AG News Classification API is running"
    }


@app.post("/predict")
def predict(request: NewsRequest):

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[0][prediction].item()

    labels = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech"
    }

    return {
        "prediction": labels[prediction],
        "confidence": confidence
    }
