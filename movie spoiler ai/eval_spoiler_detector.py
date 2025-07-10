# eval_spoiler_detector.py

import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

# 1. Prepare test data
test_data = {
    "text": [
        "The hero dies in the end.",
        "Amazing special effects.",
        "She was the villain.",
        "Loved the soundtrack."
    ],
    "label": [1, 0, 1, 0]
}
df_test = pd.DataFrame(test_data)

# 2. Load tokenizer and model from saved folder
tokenizer = BertTokenizer.from_pretrained("spoiler_model")
model = BertForSequenceClassification.from_pretrained("spoiler_model")

# 3. Dataset class (reuse same class as training)
class SpoilerDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

test_dataset = SpoilerDataset(df_test["text"].tolist(), df_test["label"].tolist())

# 4. Trainer for evaluation
training_args = TrainingArguments(
    output_dir="./spoiler_model",
    per_device_eval_batch_size=2,
    do_train=False,
    do_eval=True,
    logging_dir="./logs",
    logging_steps=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

# 5. Evaluate and print metrics
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
