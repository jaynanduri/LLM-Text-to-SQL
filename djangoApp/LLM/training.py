from transformers import TrainingArguments, Trainer
from model import tokenizer, model, preprocess_logits_for_metrics, compute_metrics
from tokenizer import dataset
from trl import DataCollatorForCompletionOnlyLM

"""
The data collator constructor expects a string or token id sequence, that
separates the response from the instructions. We’ll use
"<|im_start|>assistant\n" as the separator, since the SQL query response always
comes after this. To be precise, we’ll pass in the token id.
"""

response_template_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
collator = DataCollatorForCompletionOnlyLM(
    response_template_ids, tokenizer=tokenizer
)
hparams = {
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v0.4",
    "dataset_subset": "easy",
    "training_args": {
        "output_dir": "/content/llm_finetuning",
        "max_steps": 5000,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 4,
        "fp16": True,
        "evaluation_strategy": "steps",
        "eval_steps": 1000,
        "logging_strategy": "steps",
        "logging_steps": 100,
        "save_strategy": "steps",
        "save_steps": 1000,
        "learning_rate": 1e-5,
        "gradient_accumulation_steps": 2
    }
}

training_args = TrainingArguments(**hparams["training_args"])

model_name = hparams["model"]
trainer = Trainer(
    args=training_args,
    model=model,
    tokenizer=tokenizer,
    data_collator=collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model('./saved_model')
tokenizer.save_pretrained('./saved_model')
