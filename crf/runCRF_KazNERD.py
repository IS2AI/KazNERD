#from itertools import chain
import nltk
import sklearn
import scipy.stats
import sklearn_crfsuite
import seqeval.metrics
from seqeval.scheme import IOB2
from features import word2features
from conlleval import evaluate

use_context=True    #extract features from +-2 and +-1 context words
L1=0.1
L2=0.01
max_iter=550
period=10

#DATA PREPARATION #################################################################################
print("DATA PREPARATION STAGE ...")
train_path = "../KazNERD/IOB2_train.txt"
valid_path = "../KazNERD/IOB2_valid.txt"
test_path = "../KazNERD/IOB2_test.txt"

train_sents = list()
valid_sents = list()
test_sents = list()
with open(train_path) as train, \
        open(valid_path) as valid, \
        open(test_path) as test:

    sent = []
    for line in train: 
        line = line.strip()
        if line != "":
            sent.append((line.split()[0], line.split()[1]))
        else:
            train_sents.append(sent)
            sent = []
            
    sent = []
    for line in valid: 
        line = line.strip()
        if line != "":
            sent.append((line.split()[0], line.split()[1]))
        else:
            valid_sents.append(sent)
            sent = []
            
    sent = []
    for line in test: 
        line = line.strip()
        if line != "":
            sent.append((line.split()[0], line.split()[1]))
        else:
            test_sents.append(sent)
            sent = []
            

#FEATURE EXTRACTION ###############################################################################
print("FEATURE EXTRACTION STAGE ...")
print("Use context features: ", use_context)
def sent2features(sent):
    return [word2features(sent, i, use_context) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_valid = [sent2features(s) for s in valid_sents]
y_valid = [sent2labels(s) for s in valid_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]


#TRAIN ############################################################################################
print("TRAINING STAGE ...")
print("L1: ", L1)
print("L2: ", L2)
print("max iterations: ", max_iter)
print("period: ", period)

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=L1,
    c2=L2,
    max_iterations=max_iter,
    period=period,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

#EVALUATION #######################################################################################
print("EVALUATION STAGE ...")
y_valid_pred = crf.predict(X_valid)
y_test_pred = crf.predict(X_test)

#Evaluate using seqeval
if True:
    print("#"*50)
    print("*** seqeval ***")
    print("#"*50)
    print("VALID SET", "#"*40)
    #print("seqeval F1:", seqeval.metrics.f1_score(y_valid, y_valid_pred))
    #print(seqeval.metrics.classification_report(y_valid, y_valid_pred, digits=4))
    print(seqeval.metrics.classification_report(y_valid, y_valid_pred, digits=4, mode='strict', scheme=IOB2))

    print("TEST SET", "#"*40)
    #print("seqeval F1:", seqeval.metrics.f1_score(y_test, y_test_pred))
    #print(seqeval.metrics.classification_report(y_test, y_test_pred, digits=4))
    print(seqeval.metrics.classification_report(y_test, y_test_pred, digits=4, mode='strict', scheme=IOB2))


#Evaluate using conlleval
if False:
    print("#"*50)
    print("*** conlleval ***")
    print("#"*50)
    y_valid_all = []
    y_valid_pred_all = []

    y_test_all = []
    y_test_pred_all = []
    #Evaluate using CONLL eval metrics
    for i in y_valid:
        y_valid_all.extend(i)
        y_valid_all.extend(['O'])

    for i in y_test:
        y_test_all.extend(i)
        y_test_all.extend(['O'])

    for i in y_valid_pred:
        y_valid_pred_all.extend(i)
        y_valid_pred_all.extend(['O'])

    for i in y_test_pred:
        y_test_pred_all.extend(i)
        y_test_pred_all.extend(['O'])

    print("VALID SET", "#"*40)
    prec, rec, f1 = evaluate(y_valid_all, y_valid_pred_all, verbose=False)
    print("conlleval F1:", f1)
    evaluate(y_valid_all, y_valid_pred_all, verbose=True) 

    print("TEST SET", "#"*40)
    prec, rec, f1 = evaluate(y_test_all, y_test_pred_all, verbose=False)
    print("conlleval F1:", f1)
    evaluate(y_test_all, y_test_pred_all, verbose=True) 

exit()
