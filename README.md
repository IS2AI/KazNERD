# Kazakh Named Entity Recognition
This repository contains an open-source Kazakh named entity recognition dataset (KazNERD), named entity annotation guidelines (in Kazakh), and NER model training codes (CRF, BiLSTM-CNN-CRF, BERT and XLM-RoBERTa).
1. [KazNERD Corpus](#KazNerd)
2. [Annotation Guidelines](#guide)
3. [NER Models](#models)
    1. [CRF](#crf)
    2. [BiLSTM-CNN-CRF](#lstm)
    3. [BERT and XLM-RoBERTa](#bert)
4. [Citation](#cite)

# 1. KazNERD Corpus <a name="KazNerd"></a>
KazNERD contains 112,702 sentences, extracted from the television news text, and 136,333 annotations for 25 entity classes.
All sentences in the dataset were manually annotated by two native Kazakh-speaking linguists, supervised by an ISSAI researcher.
The IOB2 scheme was used for annotation.
The dataset, in CoNLL 2002 format, is located [here](KazNERD).


# 2. Annotation Guidelines <a name="guide"></a>
The annotation guidelines followed to build KazNERD are located [here](Annotation_guidelines). The guidelines contain rules for annotating 25 named entity classes and their examples. The guidelines are in the Kazakh language.


# 3. NER Models <a name="models"></a>
## 3.1 CRF <a name="crf"></a>
### Conda Environment Setup for CRF
The CRF-based NER model training codes are based on **Python 3.8**.
To ease the experiment replication experience, we recommend setting up a **Conda** environment. 
```bash
conda create --name knerdCRF python=3.8
conda activate knerdCRF
conda install -c anaconda nltk scikit-learn
conda install -c conda-forge sklearn-crfsuite seqeval
```

### Start CRF training
```bash
$ cd crf
$ python runCRF_KazNERD.py
```


## 3.2 BiLSTM-CNN-CRF <a name="lstm"></a>
### Conda Environment Setup for BiLSTM-CNN-CRF
The BiLSTM-CNN-CRF-based NER model training codes are based on **Python 3.8** and **PyTorch 1.7.1**.
To ease the experiment replication experience, we recommend setting up a **Conda** environment. 
```bash
conda create --name knerdLSTM python=3.8
conda activate knerdLSTM
# Check https://pytorch.org/get-started/previous-versions/#v171
# to install a PyTorch version suitable for your OS and CUDA
# or feel free to adapt the code to a newer PyTorch version
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch   # we used this version
conda install -c conda-forge tqdm seqeval
```

### Start BiLSTM-CNN-CRF training
```bash
$ cd BiLSTM_CNN_CRF
$ bash run_train_p.sh
```

## 3.3 BERT and XLM-RoBERTa <a name="bert"></a>
### Conda Environment Setup for BERT and XLM-RoBERTa
The BERT- and XLM-RoBERTa-based NER models training codes are based on **Python 3.8** and **PyTorch 1.7.1**.
To ease the experiment replication experience, we recommend setting up a **Conda** environment. 
```bash
conda create --name knerdBERT python=3.8
conda activate knerdBERT
# Check https://pytorch.org/get-started/previous-versions/#v171
# to install a PyTorch version suitable for your OS and CUDA
# or feel free to adapt the code to a newer PyTorch version
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch   # we used this version
conda install -c anaconda numpy
conda install -c conda-forge seqeval
pip install transformers
pip install datasets
```

### Start BERT training
```bash
$ cd bert
$ python run_finetune_kaznerd.py bert
```

### Start XLM-RoBERTa training
```bash
$ cd bert
$ python run_finetune_kaznerd.py roberta
```

### Tag a custom sentence
Once the training is complete, you can use the following script to tag an input sentence:
```bash
$ python run_predict_kaznerd.py model model_checkpoint 'kazakh sentence'
```
where _model_ is either 'bert' or 'roberta', _model_checkpoint_ is a path to the pre-trained model, _'kazakh sentence'_ is an input sentence.
For example:
```bash
$ python run_predict_kaznerd.py bert bert-base-multilingual-cased-finetuned-ner-6/checkpoint-705/ 'Кеше Әйгерім Әбдібекова Абайдың «Қара сөздерінің» аудиодискісін 1000 теңгеге алды.'
```

# 4. Citation <a name="cite"></a>

```bibtex
@inproceedings{yeshpanov-etal-2022-kaznerd,
    title = "{K}az{NERD}: {K}azakh Named Entity Recognition Dataset",
    author = "Yeshpanov, Rustem  and
      Khassanov, Yerbolat  and
      Varol, Huseyin Atakan",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.44",
    pages = "417--426",
    abstract = "We present the development of a dataset for Kazakh named entity recognition. The dataset was built as there is a clear need for publicly available annotated corpora in Kazakh, as well as annotation guidelines containing straightforward{---}but rigorous{---}rules and examples. The dataset annotation, based on the IOB2 scheme, was carried out on television news text by two native Kazakh speakers under the supervision of the first author. The resulting dataset contains 112,702 sentences and 136,333 annotations for 25 entity classes. State-of-the-art machine learning models to automatise Kazakh named entity recognition were also built, with the best-performing model achieving an exact match F1-score of 97.22{\%} on the test set. The annotated dataset, guidelines, and codes used to train the models are freely available for download under the CC BY 4.0 licence from https://github.com/IS2AI/KazNERD.",
}
```
