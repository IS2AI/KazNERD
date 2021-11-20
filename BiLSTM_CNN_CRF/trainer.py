# @Author : bamtercelboo
# @Datetime : 2018/8/26 8:30
# @File : trainer.py
# @Last Modify Time : 2018/8/26 8:30
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  trainer.py
    FUNCTION : None
"""

import os, pdb
import seqeval.metrics
from seqeval.scheme import IOB2
import sys
import time
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils as utils
from DataUtils.Optim import Optimizer
from DataUtils.utils import *
from DataUtils.eval_bio import entity_evalPRF_exact, entity_evalPRF_propor, entity_evalPRF_binary
from DataUtils.eval import Eval, EvalPRF
from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Train(object):
    """
        Train
    """
    def __init__(self, **kwargs):
        """
        :param kwargs:
        Args of data:
            train_iter : train batch data iterator
            dev_iter : dev batch data iterator
            test_iter : test batch data iterator
        Args of train:
            model : nn model
            config : config
        """
        print("Training Start......")
        # for k, v in kwargs.items():
        #     self.__setattr__(k, v)
        self.train_iter = kwargs["train_iter"]
        self.dev_iter = kwargs["dev_iter"]
        self.test_iter = kwargs["test_iter"]
        self.model = kwargs["model"]
        self.config = kwargs["config"]
        self.use_crf = self.config.use_crf
        self.average_batch = self.config.average_batch
        self.early_max_patience = self.config.early_max_patience
        self.optimizer = Optimizer(name=self.config.learning_algorithm, model=self.model,
                                    lr=self.config.learning_rate,
                                    weight_decay=self.config.weight_decay,
                                    grad_clip=self.config.clip_max_norm,
                                    momentum=self.config.momentum)
        self.loss_function = self._loss(learning_algorithm=self.config.learning_algorithm,
                                        label_paddingId=self.config.label_paddingId, use_crf=self.use_crf)
        print(self.optimizer)
        print(self.loss_function)
        self.best_score = Best_Result()
        self.train_eval, self.dev_eval, self.test_eval = Eval(), Eval(), Eval()
        self.train_iter_len = len(self.train_iter)

    def _loss(self, learning_algorithm, label_paddingId, use_crf=False):
        """
        :param learning_algorithm:
        :param label_paddingId:
        :param use_crf:
        :return:
        """
        if use_crf:
            loss_function = self.model.crf_layer.neg_log_likelihood_loss
            return loss_function
        elif learning_algorithm == "SGD":
            loss_function = nn.CrossEntropyLoss(ignore_index=label_paddingId, reduction="sum")
            return loss_function
        else:
            loss_function = nn.CrossEntropyLoss(ignore_index=label_paddingId, reduction="mean")
            return loss_function

    def _clip_model_norm(self, clip_max_norm_use, clip_max_norm):
        """
        :param clip_max_norm_use:  whether to use clip max norm for nn model
        :param clip_max_norm: clip max norm max values [float or None]
        :return:
        """
        if clip_max_norm_use is True:
            gclip = None if clip_max_norm == "None" else float(clip_max_norm)
            assert isinstance(gclip, float)
            utils.clip_grad_norm_(self.model.parameters(), max_norm=gclip)

    def _dynamic_lr(self, config, epoch, new_lr):
        """
        :param config:  config
        :param epoch:  epoch
        :param new_lr:  learning rate
        :return:
        """
        if config.use_lr_decay is True and epoch > config.max_patience and (
                epoch - 1) % config.max_patience == 0 and new_lr > config.min_lrate:
            new_lr = max(new_lr * config.lr_rate_decay, config.min_lrate)
            set_lrate(self.optimizer, new_lr)
        return new_lr

    def _decay_learning_rate(self, epoch, init_lr):
        """lr decay 

        Args:
            epoch: int, epoch 
            init_lr:  initial lr
        """
        lr = init_lr / (1 + self.config.lr_rate_decay * epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return self.optimizer

    def _optimizer_batch_step(self, config, backward_count):
        """
        :param config:
        :param backward_count:
        :return:
        """
        if backward_count % config.backward_batch_size == 0 or backward_count == self.train_iter_len:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _early_stop(self, epoch):
        """
        :param epoch:
        :return:
        """
        best_epoch = self.best_score.best_epoch
        if epoch > best_epoch:
            self.best_score.early_current_patience += 1
            print("Dev Has Not Promote {} / {}".format(self.best_score.early_current_patience, self.early_max_patience))
            if self.best_score.early_current_patience >= self.early_max_patience:
                print("Early Stop Train. Best Score Locate on {} Epoch.".format(self.best_score.best_epoch))
                exit()

    @staticmethod
    def _get_model_args(batch_features):
        """
        :param batch_features:  Batch Instance
        :return:
        """
        word = batch_features.word_features
        char = batch_features.char_features
        mask = word > 0
        sentence_length = batch_features.sentence_length
        tags = batch_features.label_features
        return word, char, mask, sentence_length, tags

    def _calculate_loss(self, feats, mask, tags):
        """
        Args:
            feats: size = (batch_size, seq_len, tag_size)
            mask: size = (batch_size, seq_len)
            tags: size = (batch_size, seq_len)
        """
        if not self.use_crf:
            batch_size, max_len = feats.size(0), feats.size(1)
            lstm_feats = feats.view(batch_size * max_len, -1)
            tags = tags.view(-1)
            return self.loss_function(lstm_feats, tags)
        else:
            loss_value = self.loss_function(feats, mask, tags)
        if self.average_batch:
            batch_size = feats.size(0)
            loss_value /= float(batch_size)
        return loss_value

    def train(self):
        """
        :return:
        """
        epochs = self.config.epochs
        clip_max_norm_use = self.config.clip_max_norm_use
        clip_max_norm = self.config.clip_max_norm
        new_lr = self.config.learning_rate

        for epoch in range(1, epochs + 1):
            print("\n## The {} Epoch, All {} Epochs ! ##".format(epoch, epochs))
            # new_lr = self._dynamic_lr(config=self.config, epoch=epoch, new_lr=new_lr)
            self.optimizer = self._decay_learning_rate(epoch=epoch - 1, init_lr=self.config.learning_rate)
            print("now lr is {}".format(self.optimizer.param_groups[0].get("lr")), end="")
            start_time = time.time()
            random.shuffle(self.train_iter)
            self.model.train()
            steps = 1
            backward_count = 0
            self.optimizer.zero_grad()
            for batch_count, batch_features in enumerate(self.train_iter):
                backward_count += 1
                # self.optimizer.zero_grad()
                word, char, mask, sentence_length, tags = self._get_model_args(batch_features)
                logit = self.model(word, char, sentence_length, train=True)
                loss = self._calculate_loss(logit, mask, tags)
                loss.backward()
                self._clip_model_norm(clip_max_norm_use, clip_max_norm)
                self._optimizer_batch_step(config=self.config, backward_count=backward_count)
                # self.optimizer.step()
                steps += 1
                if (steps - 1) % self.config.log_interval == 0:
                    self.getAcc(self.train_eval, batch_features, logit, self.config)
                    sys.stdout.write(
                        "\nbatch_count = [{}] , loss is {:.6f}, [TAG-ACC is {:.6f}%]".format(batch_count + 1, loss.item(), self.train_eval.acc()))
            end_time = time.time()
            print("\nTrain Time {:.3f}".format(end_time - start_time), end="")
            self.eval(model=self.model, epoch=epoch, config=self.config)
            self._model2file(model=self.model, config=self.config, epoch=epoch)
            self._early_stop(epoch=epoch)

    def eval(self, model, epoch, config):
        """
        :param model: nn model
        :param epoch:  epoch
        :param config:  config
        :return:
        """
        self.dev_eval.clear_PRF()
        eval_start_time = time.time()
        self.eval_batch(self.dev_iter, model, self.dev_eval, self.best_score, epoch, config, test=False)
        eval_end_time = time.time()
        print("Dev Time {:.3f}".format(eval_end_time - eval_start_time))

        self.test_eval.clear_PRF()
        eval_start_time = time.time()
        self.eval_batch(self.test_iter, model, self.test_eval, self.best_score, epoch, config, test=True)
        eval_end_time = time.time()
        print("Test Time {:.3f}".format(eval_end_time - eval_start_time))

    def _model2file(self, model, config, epoch):
        """
        :param model:  nn model
        :param config:  config
        :param epoch:  epoch
        :return:
        """
        if config.save_model and config.save_all_model:
            save_model_all(model, config.save_dir, config.model_name, epoch)
        elif config.save_model and config.save_best_model:
            save_best_model(model, config.save_best_model_path, config.model_name, self.best_score)
        else:
            print()

    def eval_batch(self, data_iter, model, eval_instance, best_score, epoch, config, test=False):
        """
        :param data_iter:  eval batch data iterator
        :param model: eval model
        :param eval_instance:
        :param best_score:
        :param epoch:
        :param config: config
        :param test:  whether to test
        :return: None
        """
        model.eval()
        # eval time
        eval_acc = Eval()
        eval_PRF = EvalPRF()
        gold_labels = []
        predict_labels = []
        for batch_features in data_iter:
            word, char, mask, sentence_length, tags = self._get_model_args(batch_features)
            logit = model(word, char, sentence_length, train=False)
            if self.use_crf is False:
                predict_ids = torch_max(logit)
                for id_batch in range(batch_features.batch_length):
                    inst = batch_features.inst[id_batch]
                    label_ids = predict_ids[id_batch]
                    predict_label = []
                    for id_word in range(inst.words_size):
                        predict_label.append(config.create_alphabet.label_alphabet.from_id(label_ids[id_word]))
                    gold_labels.append(inst.labels)
                    predict_labels.append(predict_label)
            else:
                path_score, best_paths = model.crf_layer(logit, mask)
                for id_batch in range(batch_features.batch_length):
                    inst = batch_features.inst[id_batch]
                    gold_labels.append(inst.labels)
                    label_ids = best_paths[id_batch].cpu().data.numpy()[:inst.words_size]
                    label = []
                    for i in label_ids:
                        # print("\n", i)
                        label.append(config.create_alphabet.label_alphabet.from_id(int(i)))
                    predict_labels.append(label)
        for p_label, g_label in zip(predict_labels, gold_labels):
            eval_PRF.evalPRF(predict_labels=p_label, gold_labels=g_label, eval=eval_instance)
        if eval_acc.gold_num == 0:
            eval_acc.gold_num = 1

        p, r, f = eval_instance.getFscore()
        f_seq = seqeval.metrics.f1_score(gold_labels, predict_labels, mode='strict')
        p_seq = seqeval.metrics.precision_score(gold_labels, predict_labels, mode='strict')
        r_seq = seqeval.metrics.recall_score(gold_labels, predict_labels, mode='strict')
        # p, r, f = entity_evalPRF_exact(gold_labels=gold_labels, predict_labels=predict_labels)
        # p, r, f = entity_evalPRF_propor(gold_labels=gold_labels, predict_labels=predict_labels)
        # p, r, f = entity_evalPRF_binary(gold_labels=gold_labels, predict_labels=predict_labels)
        test_flag = "Test"
        if test is False:
            print()
            test_flag = "Dev"
            best_score.current_dev_score = f
            if f >= best_score.best_dev_score:
                best_score.best_dev_score = f
                best_score.best_epoch = epoch
                best_score.best_test = True
                #seqeval computed scores
                best_score.best_dev_f_seq = f_seq
                best_score.best_dev_p_seq = p_seq
                best_score.best_dev_r_seq = r_seq
        if test is True and best_score.best_test is True:
            best_score.p = p
            best_score.r = r
            best_score.f = f
            #seqeval computed scores
            best_score.best_test_f_seq = f_seq
            best_score.best_test_p_seq = p_seq
            best_score.best_test_r_seq = r_seq
        print("{} eval: precision = {:.6f}%  recall = {:.6f}% , f-score = {:.6f}%,  [TAG-ACC = {:.6f}%]".format(
                test_flag, p, r, f, 0.0000))
        print("*** {} seqeval: precision = {:.6f}%  recall = {:.6f}% , f-score = {:.6f}%,  [TAG-ACC = {:.6f}%]".format(
                test_flag, p_seq, r_seq, f_seq, 0.0000))
        if test is True:
            print("The Current Best Dev F-score: {:.6f}, Locate on {} Epoch.".format(best_score.best_dev_score,
                                                                                     best_score.best_epoch))
            print("The Current Best Test Result: precision = {:.6f}%  recall = {:.6f}% , f-score = {:.6f}%".format(
                best_score.p, best_score.r, best_score.f))
            print("*** The Current Best Dev seqeval Result: precision = {:.6f}%  recall = {:.6f}% , f-score = {:.6f}%".format(
                best_score.best_dev_p_seq, best_score.best_dev_r_seq, best_score.best_dev_f_seq ))
            print("*** The Current Best Test seqeval Result: precision = {:.6f}%  recall = {:.6f}% , f-score = {:.6f}%".format(
                best_score.best_test_p_seq, best_score.best_test_r_seq, best_score.best_test_f_seq ))
        if test is True:
            best_score.best_test = False

    @staticmethod
    def getAcc(eval_acc, batch_features, logit, config):
        """
        :param eval_acc:  eval instance
        :param batch_features:  batch data feature
        :param logit:  model output
        :param config:  config
        :return:
        """
        eval_acc.clear_PRF()
        predict_ids = torch_max(logit)
        for id_batch in range(batch_features.batch_length):
            inst = batch_features.inst[id_batch]
            label_ids = predict_ids[id_batch]
            predict_label = []
            gold_lable = inst.labels
            for id_word in range(inst.words_size):
                predict_label.append(config.create_alphabet.label_alphabet.from_id(label_ids[id_word]))
            assert len(predict_label) == len(gold_lable)
            cor = 0
            for p_lable, g_lable in zip(predict_label, gold_lable):
                if p_lable == g_lable:
                    cor += 1
            eval_acc.correct_num += cor
            eval_acc.gold_num += len(gold_lable)

