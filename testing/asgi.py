# from datasets import load_dataset
# from transformers import AutoTokenizer, BartForConditionalGeneration
# from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
# import torch
# import numpy as np

# # Load the dataset
# dataset_samsum = load_dataset("samsum")

# # Initialize tokenizer and model with the fine-tuned checkpoint
# model_ckpt = "facebook/bart-base"  # Switched to a smaller model for memory efficiency
# tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# model = BartForConditionalGeneration.from_pretrained(model_ckpt)

# # Check if CUDA is available and move the model to the appropriate device
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# def preprocess_data(example):
#     inputs = tokenizer(example['dialogue'], max_length=512, truncation=True, padding="max_length", return_tensors="pt")
#     targets = tokenizer(example['summary'], max_length=128, truncation=True, padding="max_length", return_tensors="pt")
#     return {
#         'input_ids': inputs['input_ids'].squeeze(),
#         'attention_mask': inputs['attention_mask'].squeeze(),
#         'labels': targets['input_ids'].squeeze()
#     }

# # Sample a small subset of the dataset for quicker training
# def sample_dataset(dataset, fraction=0.5):
#     indices = np.random.choice(len(dataset), size=int(len(dataset) * fraction), replace=False)
#     return dataset.select(indices)

# # Sample the training and validation datasets
# train_subset = sample_dataset(dataset_samsum['train'])
# val_subset = sample_dataset(dataset_samsum['validation'])

# # Preprocess the datasets
# tokenized_train_subset = train_subset.map(preprocess_data, batched=True)
# tokenized_val_subset = val_subset.map(preprocess_data, batched=True)

# # Training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     per_device_train_batch_size=2,  # Decreased batch size
#     per_device_eval_batch_size=2,   # Decreased batch size
#     num_train_epochs=1,  # Reduced number of epochs for quick testing
#     weight_decay=0.01,
#     logging_steps=10,
#     save_steps=50,  # Save more frequently for quicker checkpoints
#     eval_steps=50,  # Evaluate more frequently
#     gradient_accumulation_steps=4,  # Increased accumulation steps
#     fp16=False,  # Disabled fp16 for better compatibility with MPS
# )

# # Data collator
# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# # Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train_subset,
#     eval_dataset=tokenized_val_subset,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
# )

# # Train the model
# trainer.train()

# # Save the model
# model.save_pretrained("trained_bart_model")
# tokenizer.save_pretrained("trained_bart_model")

# def summarize_call(call_text):
#     inputs = tokenizer(call_text, max_length=512, truncation=True, return_tensors="pt").to(device)
#     summary_ids = model.generate(inputs['input_ids'], max_length=128, num_beams=4, length_penalty=2.0, early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summary

# # Example call text
# call_text = """
# HELLO MY SWEET HEART DO YOU WANT TO GO OUT WITH ME TO MORROW NIGHT WHO 
# SWEETORT WHOS CALLING GUEESE ARE YOU KITTING ME JESSE EMILY YOUR BESTE HERE EXCUSE 
# ME YOU MAY HAVE DILED THE WRONG NUMBER THERE'S NOT A JESSE YEAR OH OH I'M SO SORRY NOT 
# AT ALL THAT'S WINE ON'T MY GOSH JESSE YOU DON'T KNOW WHAT HAPPEND TO ME A FEW MINUTES AGO 
# EMILY THIS IS JESSE'S MAM SPEAKING OH HI MISSUS VICTORIA WHERE IS JESSE SHE IS JUST GONE TO THE 
# SUPER MARKET NEAR HER HOUSE SHE WILL BE BACK RIGHT NOW WOULD YOU LIKE TO HOLD YES PLEASE JUST HOLD THE
# LINE PLEASE WHEN JESSE COMES HOME I'LL TELL HER JESSE EMILY IS CALLING YOU I TOLD HER TO HANG ON WHILE WAITING 
# FOR YOU THINK SO MUCH MOMMY ELLO MY FRIEND I JUST CAME BACK FROM THE SUPERMARQET DON'T MIRY JESSE 
# YOU WON'T BELIEVE WHAT I'LL SAY WHAT HAPPENED I CALLD YOU BUT THE WRONG NUMBER MORT INPORTANTLY THE
# STRANGER IS A MAN AND I SAID HELLO MY SWEET HARD OH REALLY HOW CAN YOU MAKE A SHAMEFUL MISTAKE LIKE 
# THAT UAH I DON'T KNOW WHERE TO HIDE I HAD AN EGG ON MY FACE COME ON GIRL IT'S NOT SERIOUS LIKE YOU THINK 
# OKE JESSE HOW ERE YOU GETTING ON WITH YOUR NEW JOM I LOVE MY JOB NOW I EASILY FALL IN LINE AND COMPLETE ALL 
# DUTEASE EACH DAY I REALLY LOVE THIS FEELING GLAD TO HEAR IT AH EMILY WHAT DID YOU CALL ME FOR OH NOTHING 
# IMPORTANT JUST TO ASK YOU IF YOU CAN COME OUT TO MORROW NIGHT OKE SO WHERE WILL WE GO SHALL WE
# HAVE DINNER TOGETHER ER JUST DREINK COFFEE IT'S UP TO YOU IF SO LET'S HAVE DINNER THEN GO TO 
# DRINK COFFEE IS THAT FINE I AGREE DINNER IS MY TREET ASOM SO DID YOU JECIDE WHICH RESTAURANT WE
# EAT AT NOT YET EAST WOOD BEER AND GRIL I JUST READ THE INTERVIEW ABOUT THAT EATING PLACE IT SEEMS 
# TO BE A GOOD RESTAURANT WHAT RESTAURANT DID YOU SAY I DID NOT HEAR CLEARLY I SAID EAST WOOD
# BEER AND GRILL EMILY I DIDN'T HEAR ANYTHING WOULD YOU MIND SPEAKING UP A BIT I THINK I'M BREAKING OUT 
# LET ME MOVE TO ANOTHER PLACE TO HAVE A BETTER CONNETION CAN I CALL BACK RIGHT NOW BECAUSE I'M HAVING 
# A BAD CONNECTION YE I SHOULD DO THAT JESSE CAN YOU HEAR ME NOW MORE CLEARLY ALITTLE BUT CAN YOU SPEAK LOUDER 
# I'M TRYING TO FIND EVERY CORNER OF MY HOUSE TO HAVE A GOOD CONNECTION CLEAR EMILY WHICH RESTAURANT DID YOU 
# SAY EAST WOOD BEER AND GRIL EAST WOOD BEER AND GRI A HAVE YOU BEEN TO THIS RESTAURANT HAW DO YOU SPELL THAT 
# PLEASE E A S T EAST WOOD OH GOT IT WHAT IS THE EXACT ADDRESS IN HILTON ANSTERDAND CENTRAL STATION DEAL
# WHAT TIME CAN WE GO AH SIX P M IS THAT OKET TOO EARLY JESSE I JUST GET HOME FROM THE OFFICE AT THAT TIME WHAT 
# ABOUT SEVEN OTOKE I'M FRA SO WHATEVER YOU WANT OKE SO WE HAVE A DAT AND EASTWOOD BEER AND GRILL AT SEVEN P M THEN 
# DRINK OFFEE DEAL ALL RIGHT AH COALD I HANG UP THE PHONE FIRST BECAUSE I'VE SOME WORK TO DO OOKE BY BY EMILY SEE TO MORROW GOOD BY JESSE SEE YOU
# """

# # Generate summary
# summary = summarize_call(call_text)
# print("Summary:", summary)



"""
ASGI config for testing project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "testing.settings")

application = get_asgi_application()