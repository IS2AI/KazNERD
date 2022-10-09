import os, sys, pdb, re
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
#gpu="0,1,2,3"
#gpu="6"
#os.environ["CUDA_VISIBLE_DEVICES"]=gpu
import numpy as np
import seqeval.metrics
from seqeval.scheme import IOB2
import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from datasets import load_dataset, load_metric

if len(sys.argv) < 4:
    print("Usage example:\n python run_predict_kaznerd.py model model_checkpoint 'kazakh sentence'\n"
            "e.g.: python run_predict_kaznerd.py bert "
            "bert-base-multilingual-cased-finetuned-ner-6/checkpoint-705/ "
            "'Кеше Әйгерім Әбдібекова Абайдың «Қара сөздерінің» аудиодискісін 1000 теңгеге алды.'")
    exit()
else:
    model_name = sys.argv[1].lower()
    if model_name != "bert" and model_name != "roberta":
        print("Incorrect model name is specified. It should be 'bert' or 'roberta'!")
        exit()
    model_checkpoint = sys.argv[2]
    input_sent = sys.argv[3]


#model_checkpoint = "bert-base-multilingual-cased-finetuned-ner-6/checkpoint-705"
#model_checkpoint = "xlm-roberta-large-finetuned-ner-5/checkpoint-14100"

labels_dict = {0:"O", 1:"B-ADAGE", 2:"I-ADAGE", 3:"B-ART", 4:"I-ART", 5:"B-CARDINAL",
        6:"I-CARDINAL", 7:"B-CONTACT", 8:"I-CONTACT", 9:"B-DATE", 10:"I-DATE", 11:"B-DISEASE",
        12:"I-DISEASE", 13:"B-EVENT", 14:"I-EVENT", 15:"B-FACILITY", 16:"I-FACILITY",
        17:"B-GPE", 18:"I-GPE", 19:"B-LANGUAGE", 20:"I-LANGUAGE", 21:"B-LAW", 22:"I-LAW",
        23:"B-LOCATION", 24:"I-LOCATION", 25:"B-MISCELLANEOUS", 26:"I-MISCELLANEOUS",
        27:"B-MONEY", 28:"I-MONEY", 29:"B-NON_HUMAN", 30:"I-NON_HUMAN", 31:"B-NORP",
        32:"I-NORP", 33:"B-ORDINAL", 34:"I-ORDINAL", 35:"B-ORGANISATION", 36:"I-ORGANISATION",
        37:"B-PERCENTAGE", 38:"I-PERCENTAGE", 39:"B-PERSON", 40:"I-PERSON", 41:"B-POSITION",
        42:"I-POSITION", 43:"B-PRODUCT", 44:"I-PRODUCT", 45:"B-PROJECT", 46:"I-PROJECT",
        47:"B-QUANTITY", 48:"I-QUANTITY", 49:"B-TIME", 50:"I-TIME"}

#Tokenize input sentence for BERT
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

tokenized_inputs = tokenizer(input_sent, return_tensors="pt")
#tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][0])

#Load model
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

#Predict
output = model(**tokenized_inputs)
predictions = np.argmax(output.logits.detach().numpy(), axis=2)

#Convert label IDs to label names
word_ids = tokenized_inputs.word_ids(batch_index=0)
previous_word_idx = None
labels = []
for i, p in zip(word_ids, predictions[0]):
    # Special tokens have a word id that is None. We set the label to -100 so they are
    # automatically ignored in the loss function.
    if i is None or i == previous_word_idx:
        continue
    elif i != previous_word_idx:
        labels.append(labels_dict[p])
    previous_word_idx = i

#Print tokens and predicted labels
if model_name == "roberta":
    input_sent_tokens = tokenizer.decode(tokenized_inputs['input_ids'][0], skip_special_tokens=True).split()
else:
    input_sent_tokens = re.findall(r"[\w’]+|[-.,#?!)(\]\[;:–—\"«№»/%&']", input_sent)
assert len(input_sent_tokens) == len(labels), "Mismatch between input token and label sizes!"
for t,l in zip(input_sent_tokens, labels):
    print(t,l)

exit()
