## Config ##

- Use `ConfigParser` to config parameter  
	- `from configparser import ConfigParser`  .
	- Detail see `config.py` and `config.cfg`, please.  

- Following is `config.cfg` Parameter details.

- [Embed]  
	- `pretrained_embed` (True or False) ------ whether to use pretrained embedding.
	- `zeros`(True or False) ------ OOV by zeros .
	- `avg`(True or False) ------ OOV by avg .
	- `uniform`(True or False) ------ OOV by uniform .
	- `nnembed`(True or False) ------ OOV by nn.Embedding .
	- `pretrained_embed_file` --- pre train path

- [Data]  
	- `train-file/dev-file/test-file`(path)  ------ train/dev/test data path(`Data`).
	- `min_freq` (integer number) ------ The smallest Word frequency when build vocab.
	- `max_count ` (integer number) ------ The maximum instance for `debug`.
	- `shuffle/epochs-shuffle`(True or False) ------ shuffle data .

- [Save]
	- `save_pkl` (True or False) ------ save pkl file for test.
	- `pkl_directory` (path) ------ save pkl directory path.
	- `pkl_data` (path) ------ save pkl data path.
	- `pkl_alphabet` (path) ------ save pkl alphabet path.
	- `pkl_iter` (path) ------ save pkl batch iterator path.
	- `pkl_embed` (path) ------ save pkl pre-train embedding path.
	- `save_dict` (True or False) ------ save dict to file.
	- `dict_directory ` (path) ------ save dict directory path.
	- `word_dict ` (path) ------ save word dict path.
	- `label_dict ` (path) ------ save label dict path.
	- `save_model ` (True or False) ------ save model to file.
	- `save_all_model` (True or False) ------ save all model to file.
	- `save_best_model` (True or False) ------ save best model to file.
	- `model_name ` (str) ------ model name.
	- `rm_model` (True or False) ------ remove model to save space(now not use).

- [Model]
	- `model-bilstm` (True or False) ------ Bilstm model.
	- `model-***`(True or False) ------ *** model.
	- `use_char `(True or False) ------ CNN feature
	- `use_crf `(True or False) ------ CRF feature
	- `lstm-layers` (integer) ------ number layers of lstm.
	- `embed-dim` (integer) ------ embedding dim = pre-trained embedding dim.
	- `embed-finetune` (True or False) ------ word embedding finetune or no-finetune.
	- `lstm-hiddens` (integer) ------numbers of lstm hidden.
	- `dropout-emb/dropout `(float) ------ dropout for prevent overfitting.
	- `windows-size` (integer) ------ Context window feature size.
	- `max_char_len` (integer) ------ The maximum length of a word.
	- `char_dim` (integer) ------ The conv char dimension.
	- `conv_filter_sizes` (str(3,4,5)) ------ conv filter sizes split by a comma in English.
	- `conv_filter_nums` (str(30,40,50)) ------ conv filter nums split by a comma in English.

- [Optimizer]
	- `adam` (True or False) ------ `torch.optim.Adam`
	- `sgd` (True or False)  ------ `torch.optim.SGD`
	- `learning-rate`(float) ------ learning rate.
	- `weight-decay` (float) ------ L2.
	- `momentum ` (float) ------ SGD momentum.
	- `clip_max_norm_use` (True or False) ------ use util.clip.
	- `clip-max-norm` (Integer number) ------ clip-max-norm value.
	- `use_lr_decay` (True or False) ------ use lr decay.
	- `lr_rate_decay`(float) ------ lr decay value.
	- `min_lrate `(float) ------ minimum lr.
	- `max_patience `(Integer) ------ patience to decay.

- [Train]
	- `num-threads` (Integer) ------ threads.  
	- `epochs` (Integer) ------ maximum train epochs
	- `early_max_patience` (Integer) ------ maximum dev no improvment times for early stop.
 	- `backward_batch_size` (Integer) ------ multiple-batch to update parameters.
	- `batch-size/dev-batch-size/test-batch-size` (Integer) ------ number of batch
	- `log-interval`(Integer) ------ steps of print log.