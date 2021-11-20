import os, sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
#gpu="0,1,2,3"
gpu="0"
os.environ["CUDA_VISIBLE_DEVICES"]=gpu
import numpy as np
import seqeval.metrics
from seqeval.scheme import IOB2
import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from datasets import load_dataset, load_metric

if len(sys.argv) < 2 or sys.argv[1].lower() == 'bert':
    print("Using BERT model")
    model_checkpoint = "bert-base-multilingual-cased" 
elif sys.argv[1].lower() == 'roberta':
    print("Using XLM-RoBERTa model")
    model_checkpoint = "xlm-roberta-large"
else:
    print("Usage example:\n python run_finetune_kaznerd.py bert")
    exit()


#print(transformers.__version__)
batch_size = 64
learning_rate = 1e-5
weight_decay = 0.001
epochs = 10
warmup_steps = 800
seed = 1


def tokenize_and_align_labels(examples, tokenizer, task, label_all_tokens=False):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True,
                        is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are
            # automatically ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or
                # -100, depending on the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                            for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                        for prediction, label in zip(predictions, labels)]

    #computes micro average
    results = metric.compute(predictions=true_predictions, references=true_labels,
                scheme="IOB2", mode="strict")
    return {"precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]}


task = "ner"
datasets = load_dataset("kaznerd.py")
label_list = datasets["train"].features[f"{task}_tags"].feature.names


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
data_collator = DataCollatorForTokenClassification(tokenizer)

metric = load_metric("seqeval")

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint,
            num_labels=len(label_list))
model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(f"{model_name}-finetuned-{task}-{gpu}",
                            overwrite_output_dir=True,
                            evaluation_strategy="epoch",
                            per_device_train_batch_size=batch_size,
                            per_device_eval_batch_size=batch_size,
                            learning_rate=learning_rate,
                            num_train_epochs=epochs,
                            warmup_steps=warmup_steps,
                            weight_decay=weight_decay,
                            save_strategy="no",
                            seed=seed,
                            push_to_hub=False)

tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True,
        fn_kwargs={"tokenizer":tokenizer,"task":task})

trainer = Trainer(model, args, 
                  data_collator=data_collator,
                  train_dataset=tokenized_datasets["train"],
                  eval_dataset=tokenized_datasets["validation"],
                  tokenizer=tokenizer,
                  compute_metrics=compute_metrics)

trainer.train()
#trainer.evaluate()

#################################################################################################
#Evaluate validation set
print("#"*100)
predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels)]
true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100]
               for prediction, label in zip(predictions, labels)]

results = metric.compute(predictions=true_predictions, references=true_labels, scheme="IOB2",
            mode="strict")
print("\nValidation: Overall F1", results["overall_f1"])
print("Validation: Total number of sentences:",len(true_labels))
print("Validation: Total number of tokens:", sum([len(sent) for sent in true_labels]))
print("Validation: seqeval based results")
print(seqeval.metrics.classification_report(true_labels, true_predictions, digits=4, mode='strict',
    scheme=IOB2))
#################################################################################################
#Evaluate test set
print("#"*100)
predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels)]
true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100]
               for prediction, label in zip(predictions, labels)]

results = metric.compute(predictions=true_predictions, references=true_labels, scheme="IOB2",
            mode="strict")
print("\nTest: Overall F1", results["overall_f1"])
print("Test: Total number of sentences:",len(true_labels))
print("Test: Total number of tokens:", sum([len(sent) for sent in true_labels]))
print("Test: seqeval based results")
print(seqeval.metrics.classification_report(true_labels, true_predictions, digits=4, mode='strict',
    scheme=IOB2))
print("#"*100)
