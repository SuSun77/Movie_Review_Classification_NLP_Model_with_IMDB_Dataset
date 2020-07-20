import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# YOUR IMPLEMENTATION

# helper function to put each sample into the corresponding positive list or negative list
def gen_set0():
    for pos in pos_sample0:
        with open(data_path0 + "/train/pos/" + pos, 'r', encoding='utf-8') as f:
            pos_set0.append(f.read())
    for neg in neg_sample0:
        with open(data_path0 + "/train/neg/" + neg, 'r', encoding='utf-8') as f:
            neg_set0.append(f.read())

def gen_set():
    for pos in pos_sample:
        with open(data_path + "/test/pos/" + pos, 'r', encoding='utf-8') as f:
            pos_set.append(f.read())
    for neg in neg_sample:
        with open(data_path + "/test/neg/" + neg, 'r', encoding='utf-8') as f:
            neg_set.append(f.read())

if __name__ == "__main__":
    # 1. Load your saved model
    model = load_model('./models/20862738_NLP_model.h5')

    # 2. Load your testing data

    # the same preprocessing procedure as the training set
    data_path = "./data/aclImdb"
    pos_sample = os.listdir(data_path + "/test/pos")
    neg_sample = os.listdir(data_path + "/test/neg")
    pos_set = []
    neg_set = []
    gen_set()
    X_orig = np.array(pos_set + neg_set)
    Y_pos = np.ones((len(pos_set)))
    Y_neg = np.zeros((len(neg_set)))
    Y_test = np.concatenate((Y_pos, Y_neg))

    # find the tokenizer
    # fitting the same tokenizer as the training set to testing set
    data_path0 = "./data/aclImdb"
    pos_sample0 = os.listdir(data_path0 + "/train/pos")
    neg_sample0 = os.listdir(data_path0 + "/train/neg")
    pos_set0 = []
    neg_set0 = []
    gen_set0()
    X_orig0 = np.array(pos_set0 + neg_set0)

    # vectorize testing set
    vocab_size = 20000
    max_len = 100
    token = Tokenizer(vocab_size)
    token.fit_on_texts(X_orig0)
    X_test_seq = token.texts_to_sequences(X_orig)
    X_test = pad_sequences(X_test_seq, maxlen=max_len)


    # 3. Run prediction on the test data and print the test accuracy
    loss, acc = model.evaluate(X_test, Y_test)
    print("Testing accuracy is ", acc)

