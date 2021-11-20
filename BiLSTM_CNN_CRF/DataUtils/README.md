## Role of document in `DataUtils` directory ##
- DataUtils
	-  `Alphabet.py`  ------ Build vocab by train data or dev/test data

	- `Batch_Iterator.py` ------ Build batch and iterator for train/dev/test data, get train/dev/test iterator

	- `Common.py` ------ The file contains some common attribute, like random seeds, padding, unk and others

	- `eval.py` ------ The file is a eval script, For calculate F-score, recall, precision. And decode model result for NER，support `BMES, BIO`  label.(Reference: https://github.com/yunan4nlp)

	- `eval_bio.py` ------ The role is same with the script of  `eval.py` ,  It is contains `eval_type = exact, binary, propor`, also support ` BMES, BIO`  label.(Reference: https://github.com/Joyce94)

	- `Load_Pretrained_Embed.py`  ------ Loading Pre-trained word embedding( `glove` or `word2vec` ), now has two way: ose is `oov` (out of vocabulary) use zero word embedding, second is `oov` use average word embedding, will add another one is `oov` use random word embedding.

	- `Load_Pretrained_Embed.py`  ------ Loading Pre-trained word embedding( `glove` or `word2vec` ), now has two way: ose is `oov` (out of vocabulary) use zero word embedding, second is `oov` use average word embedding, will add another one is `oov` use random word embedding.

	- `Embed.py`  ------ overwrite `Load_Pretrained_Embed.py`, `zeros，avg, uniform, nn.Embedding for OOV`.

	-  `Embed_From_Pretrained.py` ------ `nn.Embedding()` from pre-trained word embedding, build a big vocabulary. It can use in `No-Finetune` word embedding.

	-  `Optim.py` ------ Encapsulate the `optimizer`.

	-  `Pickle.py` ------ Encapsulate the `pickle`.

	-  `tagSchemeConverter.py` ------ convert NER label （Reference: https://github.com/jiesutd/NCRFpp/blob/master/utils/tagSchemeConverter.py）

	-  `utils.py` ------ common function.

	-  `mainHelp.py` ------ main help file.
