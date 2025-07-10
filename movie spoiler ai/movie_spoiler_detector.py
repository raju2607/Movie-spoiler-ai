import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer and model (same as training)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

model.eval()  # set to evaluation mode

def predict_spoiler(text):
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    
    return "Spoiler" if prediction == 1 else "Not Spoiler"

# Test the model
test_sentences = [
    "The hero dies at the end.",
    "This movie was full of funny moments!",
    "She was actually the villain all along.",
    "Amazing soundtrack and visuals."
]

for sentence in test_sentences:
    result = predict_spoiler(sentence)
    print(f"Text: {sentence}\nPrediction: {result}\n")
