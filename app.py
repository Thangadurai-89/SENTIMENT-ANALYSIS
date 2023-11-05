from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flask import Flask, request, render_template
import torch

app = Flask(__name__)

# Load the model and tokenizer
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        # Tokenize the user's input
        inputs = tokenizer(user_input, padding=True, truncation=True, return_tensors="pt")
        # Perform inference
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment = model.config.id2label[predictions.argmax().item()]
        return render_template("result.html", user_input=user_input, sentiment=sentiment)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

