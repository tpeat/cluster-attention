import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
import evaluate


dataset = load_dataset("opus_books", "en-fr")
books = dataset["train"].train_test_split(test_size=0.2)
source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
clustered_weights_path = "modified_checkpoint_per_cluster_80"
model = AutoModelForSeq2SeqLM.from_pretrained(clustered_weights_path)
model.generation_config.max_new_tokens = 128

def preprocess_function(examples):
  inputs = [prefix + example[source_lang] for example in examples["translation"]]
  targets = [example[target_lang] for example in examples["translation"]]
  model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
  return model_inputs


tokenized_books = books.map(preprocess_function, batched=True)    
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
metric = evaluate.load("sacrebleu")


def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


def train():
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="baseline_results_inc_max_len",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=True
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_books["train"],
        eval_dataset=tokenized_books["test"],
        processing_class=tokenizer.__class__,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


def evaluate():
    training_args = Seq2SeqTrainingArguments(
        output_dir="evaluation_results",
        per_device_eval_batch_size=128,
        predict_with_generate=True,
        fp16=True,
        do_train=False,  # Disable training
        do_eval=True     # Enable evaluation
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_books["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

if __name__ == "__main__":
    #train()
    evaluate()