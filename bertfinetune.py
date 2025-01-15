import json
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch

# Load the dataset
with open(r'C:\Users\DELL\Desktop\finetune_fitness_data.json', 'r') as f:
    data = json.load(f)

# Prepare data for fine-tuning
questions = [entry['question'] for entry in data]
answers = [entry['answer'] for entry in data]
data_df = pd.DataFrame({'question': questions, 'answer': answers})

# Encode the answers as labels (convert to numbers)
data_df['label'] = data_df['answer'].astype('category').cat.codes
answer_mapping = dict(enumerate(data_df['answer'].astype('category').cat.categories))

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data_df['question'].tolist(), data_df['label'].tolist(), test_size=0.2, random_state=42
)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Custom dataset class
class FitnessDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Create datasets
train_dataset = FitnessDataset(train_encodings, train_labels)
val_dataset = FitnessDataset(val_encodings, val_labels)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(answer_mapping))

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Fine-tune the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./fitness_qa_model')
tokenizer.save_pretrained('./fitness_qa_model')

# Save the label mapping
with open('./fitness_qa_model/label_mapping.json', 'w') as f:
    json.dump(answer_mapping, f)

print("Model and tokenizer saved successfully.")
