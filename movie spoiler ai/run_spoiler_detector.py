import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the saved model and tokenizer from the folder 'spoiler_model'
model = BertForSequenceClassification.from_pretrained("spoiler_model")
tokenizer = BertTokenizer.from_pretrained("spoiler_model")

model.eval()  # Set model to evaluation mode

def predict_spoiler(text):
    # Tokenize and encode the input text with truncation and padding
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        
    return "Spoiler" if predicted_class == 1 else "Not Spoiler"

print("Enter movie text (type 'exit' to quit):")
while True:
    user_input = input()
    if user_input.lower() == 'exit':
        break
    prediction = predict_spoiler(user_input)
    print(f"Prediction: {prediction}")
