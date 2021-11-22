# Kazakh Named Entity Recognition
This repository contains an open-source Kazakh named entity recognition dataset (KazNERD), named entity annotation guidelines (in Kazakh), and NER model training codes (CRF, BiLSTM-CNN-CRF, BERT and XLM-RoBERTa).


## KazNERD Corpus
KazNERD contains 112,702 sentences, extracted from the television news text, and 136,333 annotations for 25 entity classes.
All sentences in the dataset were manually annotated by two native Kazakh-speaking linguists, supervised by ISSAI's scientist.
The IOB2 scheme was used for annotation.
The dataset, in CoNLL 2002 format, is located [here](KazNERD).


## Annotation Guideline
The annotation guideline for 25 named entity classes is located here.
The guideline is written in Kazakh language.


## NER Models

### Setup Conda Environment
The NER training codes are based on **PyTorch 1.7.1** and **Python 3.8**.
To ease the experiment replication experience, we recommend to setup **Conda** environment. 

Create conda environment and install dependencies as follows:
```
$ conda create --name <env> --file requirements.txt
```
where `<env>` will be used as a name of created conda environment, replace _<env>_ with any string (e.g., *kaznerd*).
  
### Run CRF 
```bash
$ cd crf
$ python runCRF_KazNERD.py
```

### Run BiLSTM-CNN-CRF
```bash
$ cd BiLSTM_CNN_CRF
$ bash run_train_p.sh
```

### Run BERT
```bash
$ cd bert
$ python run_finetune_kaznerd.py bert
```

### Run XLM-RoBERTa
```bash
$ cd bert
$ python run_finetune_kaznerd.py roberta
```

## Citation

```bibtex
@inproceedings{
}
```
