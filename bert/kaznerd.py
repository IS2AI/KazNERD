# coding=utf-8
"""Kazakh Named Entity Recognition Dataset (KazNERD)"""

import datasets


logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = "Kazakh Named Entity Recognition Dataset (KazNERD)"
_URL = "../KazNERD/"
_TRAINING_FILE = "IOB2_train.txt"
_DEV_FILE = "IOB2_valid.txt"
_TEST_FILE = "IOB2_test.txt"


class KazNERDConfig(datasets.BuilderConfig):
    """BuilderConfig for KazNERD"""

    def __init__(self, **kwargs):
        """BuilderConfig for KazNERD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(KazNERDConfig, self).__init__(**kwargs)


class KazNERD(datasets.GeneratorBasedBuilder):
    """KazNERD dataset."""

    BUILDER_CONFIGS = [
        KazNERDConfig(name="kaznerd", version=datasets.Version("1.0.0"), description="KazNERD dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            supervised_keys=None,
            homepage="https://www.https://issai.nu.edu.kz/",
            #citation=_CITATION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-ADAGE",
                                "I-ADAGE",
                                "B-ART",
                                "I-ART",
                                "B-CARDINAL",
                                "I-CARDINAL",
                                "B-CONTACT",
                                "I-CONTACT",
                                "B-DATE",
                                "I-DATE",
                                "B-DISEASE",
                                "I-DISEASE",
                                "B-EVENT",
                                "I-EVENT",
                                "B-FACILITY",
                                "I-FACILITY",
                                "B-GPE",
                                "I-GPE",
                                "B-LANGUAGE",
                                "I-LANGUAGE",
                                "B-LAW",
                                "I-LAW",
                                "B-LOCATION",
                                "I-LOCATION",
                                "B-MISCELLANEOUS",
                                "I-MISCELLANEOUS",
                                "B-MONEY",
                                "I-MONEY",
                                "B-NON_HUMAN",
                                "I-NON_HUMAN",
                                "B-NORP",
                                "I-NORP",
                                "B-ORDINAL",
                                "I-ORDINAL",
                                "B-ORGANISATION",
                                "I-ORGANISATION",
                                "B-PERCENTAGE",
                                "I-PERCENTAGE",
                                "B-PERSON",
                                "I-PERSON",
                                "B-POSITION",
                                "I-POSITION",
                                "B-PRODUCT",
                                "I-PRODUCT",
                                "B-PROJECT",
                                "I-PROJECT",
                                "B-QUANTITY",
                                "I-QUANTITY",
                                "B-TIME",
                                "I-TIME",
                            ]
                        )
                    )
                }
            )
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            "dev": f"{_URL}{_DEV_FILE}",
            "test": f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line.strip() == "": 
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # kaznerd tokens are space separated
                    splits = line.split()
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
            # the last line in kaznerd's train, valid, and test sets is empty line
            # if you removed that empty line uncomment the following code to read the last example
            #yield guid, {
            #    "id": str(guid),
            #    "tokens": tokens,
            #    "ner_tags": ner_tags,
            #}
