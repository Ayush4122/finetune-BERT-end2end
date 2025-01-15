import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import json

# Load model, tokenizer, and label mapping
model_path = './fitness_qa_model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
with open(f'{model_path}/label_mapping.json', 'r') as f:
    label_mapping = json.load(f)
label_mapping = {int(k): v for k, v in label_mapping.items()}  # Convert keys to int

# Function to get the answer
def get_answer(question):
    inputs = tokenizer(question, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return label_mapping[predicted_label]

# Streamlit interface
st.title("Fitness Q&A Chatbot")
st.write("Ask me any fitness-related question, and I'll provide an answer!")

# User input
user_question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if user_question.strip():
        answer = get_answer(user_question)
        st.success(f"**Answer:** {answer}")
    else:
        st.warning("Please enter a question.")
