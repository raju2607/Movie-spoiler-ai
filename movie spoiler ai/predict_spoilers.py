import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load saved model and tokenizer
model = BertForSequenceClassification.from_pretrained("spoiler_model")
tokenizer = BertTokenizer.from_pretrained("spoiler_model")

# Put model in evaluation mode
model.eval()

def predict_spoiler(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return "Spoiler" if predicted_class == 1 else "Not Spoiler"

# List of movie texts you want to check
texts = [
    "The hero dies at the end.",
    "The cinematography was stunning.",
    "She was actually the villain.",
    "Amazing soundtrack!"
]

for text in texts:
    prediction = predict_spoiler(text)
    print(f"Text: {text}\nPrediction: {prediction}\n")
