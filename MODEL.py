from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
checkpoint="distilbert-base-uncased-finetuned-sst-2-english"



# Initialize the tokenizer to
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

# Tokenize and prepare the inputs with TensorFlow tensors
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
#model classfied for sequence
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits)


predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
print(model.config.id2label)