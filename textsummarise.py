from transformers import pipeline, BartTokenizer, BartForConditionalGeneration

def generate_summary(text):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example usage:
input_text = "Input the text to summarise."
summary_result = generate_summary(input_text)
print(summary_result)
