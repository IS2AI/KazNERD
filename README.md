# Kazakh Named Entity Recognition
This repository contains an open-source Kazakh named entity recognition dataset (KazNERD), named entity annotation guidelines (in Kazakh), and NER model training codes (CRF, BiLSTM-CNN-CRF, BERT and XLM-RoBERTa).
1. [KazNERD Corpus](#KazNerd)
2. [Annotation Guideline](#guide)
3. [NER Models](#models)
    1. [CRF](#crf)
    2. [BiLSTM-CNN-CRF](#lstm)
    3. [BERT and XLM-RoBERTa](#bert)
4. [Citation](#cite)

# 1. KazNERD Corpus <a name="KazNerd"></a>
KazNERD contains 112,702 sentences, extracted from the television news text, and 136,333 annotations for 25 entity classes.
All sentences in the dataset were manually annotated by two native Kazakh-speaking linguists, supervised by ISSAI's scientist.
The IOB2 scheme was used for annotation.
The dataset, in CoNLL 2002 format, is located [here](KazNERD).


# 2. Annotation Guideline <a name="guide"></a>
The annotation guideline for 25 named entity classes is located here.
The guideline is written in Kazakh language.


# 3. NER Models <a name="models"></a>
## 3.1 CRF <a name="crf"></a>
### Setup Conda Environment for CRF
The CRF-based NER model training codes are based on **Python 3.8**.
To ease the experiment replication experience, we recommend to setup **Conda** environment. 
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
### Setup Conda Environment for BiLSTM-CNN-CRF
The BiLSTM-CNN-CRF-based NER model training codes are based on **Python 3.8** and **PyTorch 1.7.1**.
To ease the experiment replication experience, we recommend to setup **Conda** environment. 
```bash
conda create --name knerdLSTM python=3.8
conda activate knerdLSTM
# Check https://pytorch.org/get-started/previous-versions/#v171
# to install a PyTorch version suitable for your OS and CUDA
# or feel free to adapt the code to newer PyTorch version
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch   # we used this version
conda install -c conda-forge tqdm seqeval
```

### Start BiLSTM-CNN-CRF training
```bash
$ cd BiLSTM_CNN_CRF
$ bash run_train_p.sh
```

## 3.3 BERT and XLM-RoBERTa <a name="bert"></a>
### Setup Conda Environment for BERT and XLM-RoBERTa
The BERT- and XLM-RoBERTa-based NER models training codes are based on **Python 3.8** and **PyTorch 1.7.1**.
To ease the experiment replication experience, we recommend to setup **Conda** environment. 
```bash
conda create --name knerdBERT python=3.8
conda activate knerdBERT
# Check https://pytorch.org/get-started/previous-versions/#v171
# to install a PyTorch version suitable for your OS and CUDA
# or feel free to adapt the code to newer PyTorch version
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

# 4. Citation <a name="cite"></a>

```bibtex
@inproceedings{
}
```
