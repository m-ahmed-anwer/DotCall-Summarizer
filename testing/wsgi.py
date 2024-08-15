# from datasets import load_dataset
# from transformers import AutoTokenizer, BartForConditionalGeneration
# from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
# import torch
# import numpy as np
# 
# # Load the dataset
# dataset_samsum = load_dataset("samsum")
# 
# # Initialize tokenizer and model
# model_ckpt = "facebook/bart-base"
# tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# model = BartForConditionalGeneration.from_pretrained(model_ckpt)
# 
# # Check if CUDA is available and move the model to the appropriate device
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
# 
# def preprocess_data(example):
#     inputs = tokenizer(example['dialogue'], max_length=512, truncation=True, padding="max_length", return_tensors="pt")
#     targets = tokenizer(example['summary'], max_length=128, truncation=True, padding="max_length", return_tensors="pt")
#     return {
#         'input_ids': inputs['input_ids'].squeeze(),
#         'attention_mask': inputs['attention_mask'].squeeze(),
#         'labels': targets['input_ids'].squeeze()
#     }
# 
# # Sample 50% of the dataset
# def sample_dataset(dataset, fraction=0.5):
#     indices = np.random.choice(len(dataset), size=int(len(dataset) * fraction), replace=False)
#     return dataset.select(indices)
# 
# # Sample the training and validation datasets
# train_subset = sample_dataset(dataset_samsum['train'])
# val_subset = sample_dataset(dataset_samsum['validation'])
# 
# # Preprocess the datasets
# tokenized_train_subset = train_subset.map(preprocess_data, batched=True)
# tokenized_val_subset = val_subset.map(preprocess_data, batched=True)
# 
# # Training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     per_device_train_batch_size=4,  # Increase batch size for faster training
#     per_device_eval_batch_size=4,   # Increase batch size for evaluation
#     num_train_epochs=3,
#     weight_decay=0.01,
#     logging_steps=10,
#     save_steps=100,  # Save more frequently for quicker checkpoints
#     eval_steps=100,  # Evaluate more frequently
#     gradient_accumulation_steps=2,  # Reduce to increase training speed
#     fp16=torch.cuda.is_available(),
# )
# 
# # Data collator
# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
# 
# # Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train_subset,
#     eval_dataset=tokenized_val_subset,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
# )
# 
# # Train the model
# trainer.train()
# 
# # Save the model
# model.save_pretrained("trained_bart_model")
# tokenizer.save_pretrained("trained_bart_model")
# 
# # Load the trained model
# model = BartForConditionalGeneration.from_pretrained("trained_bart_model").to(device)
# tokenizer = AutoTokenizer.from_pretrained("trained_bart_model")
# 
# def summarize_call(call_text):
#     inputs = tokenizer(call_text, max_length=512, truncation=True, return_tensors="pt").to(device)
#     summary_ids = model.generate(inputs['input_ids'], max_length=128, num_beams=4, length_penalty=2.0, early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summary
# 
# # Example call text
# call_text = """
# John: Hey, how are you doing?
# Mary: I'm good, thanks! How about you?
# John: I'm doing well, just been busy with work.
# Mary: I can imagine. Did you finish that project you were working on?
# John: Yes, finally! It was quite challenging but I managed to get it done.
# Mary: That's great to hear. Do you have any plans for the weekend?
# John: Not yet. Maybe just relax and catch up on some sleep.
# Mary: Sounds like a good plan. We should meet up sometime next week.
# John: Definitely. Let's arrange something.
# """
# 
# # Generate summary
# summary = summarize_call(call_text)
# print("Summary:", summary)


"""
WSGI config for testing project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "testing.settings")

application = get_wsgi_application()