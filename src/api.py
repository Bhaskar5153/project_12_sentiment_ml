from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Sentiment/Emotion Classifier")

# Load trained model + vectorizer
model = joblib.load("models/logreg.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    X = vectorizer.transform([input.text])
    pred = model.predict(X)[0]
    return {"text": input.text, "predicted_emotion": pred}