# KazNERD
An open-sournce Kazakh named entity recognition dataset.


## Setup Conda Environment
The code is based on **PyTorch 1.7.1** and **Python 3.8**.  
Install requirements:
```
$ conda create --name <env> --file requirements.txt
```

## Running NER models
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
