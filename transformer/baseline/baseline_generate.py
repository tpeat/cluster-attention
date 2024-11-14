from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
def main():

    # Load the model and tokenizer from checkpoint
    checkpoint_path = "baseline_results/checkpoint-31775"
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Input text for generation
    input_text = "translate English to French: I am going to get my nails done today."
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate text
    outputs = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)

    # Decode and print the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
    
if __name__ == '__main__':
    main()
