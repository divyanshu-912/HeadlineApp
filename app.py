from flask import Flask, render_template, request
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)

# Load the trained model and tokenizer
MODEL_PATH = "Best_headline_model"
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def generate_headline(article):
    input_text = "summarize: " + article
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50, num_beams=5, early_stopping=True)
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

@app.route("/", methods=["GET", "POST"])
def home():
    headline = None
    if request.method == "POST":
        article = request.form["article"]
        if article.strip():
            headline = generate_headline(article)
    return render_template("index.html", headline=headline)

if __name__ == "__main__":
    app.run(debug=True)
