import pandas as pd
from transformers import AutoTokenizer

# Step 1: Create sample movie spoiler data
data = {
    "text": [
        "Bruce Willis was dead the whole time.",
        "This movie had great visual effects!",
        "The ending where he sacrifices himself was sad.",
        "The acting and direction were amazing."
    ],
    "label": [1, 0, 1, 0]  # 1 = spoiler, 0 = not spoiler
}

# Step 2: Create DataFrame
df = pd.DataFrame(data)

# Step 3: Save to CSV (optional)
df.to_csv("spoiler_data.csv", index=False)

# Step 4: Print original DataFrame
print("Original Data:")
print(df)

# Step 5: Load tokenizer (BERT)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Step 6: Function to tokenize the text
def preprocess_texts(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

# Step 7: Tokenize all review texts
encodings = preprocess_texts(df["text"].tolist())

# Step 8: Print one example of tokenized output
print("\nSample tokenized input IDs:")
print(encodings['input_ids'][0])
