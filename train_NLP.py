import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models
from keras.layers import Dense,Flatten,Embedding

# YOUR IMPLEMENTATION

# helper function to put each sample into the corresponding positive list or negative list
def gen_set():
    for pos in pos_sample:
        with open(data_path + "/train/pos/" + pos, 'r', encoding='utf-8') as f:
            pos_set.append(f.read())
    for neg in neg_sample:
        with open(data_path + "/train/neg/" + neg, 'r', encoding='utf-8') as f:
            neg_set.append(f.read())

if __name__ == "__main__":
    # 1. load your training data

    # load original dataset from /data/aclImdb folder
    data_path = "./data/aclImdb"
    pos_sample = os.listdir(data_path + "/train/pos")
    neg_sample = os.listdir(data_path + "/train/neg")

    # create an empty list to store samples
    pos_set = []
    neg_set = []

    # call the helper function
    gen_set()

    # original training set
    X_orig = np.array(pos_set + neg_set)

    # create labels for positive samples (1)
    Y_pos = np.ones((len(pos_set)))

    # create labels for negative samples (0)
    Y_neg = np.zeros((len(neg_set)))

    # concatenate positive and negative labels
    Y_orig = np.concatenate((Y_pos, Y_neg))

    # choose 20000 as the vocabulary size (retain the first 20000 words with the highest frequency)
    vocab_size = 20000

    # choose 100 as the max length size (retain maximum of 100 words for every sample)
    max_len = 100

    # vectorize sentences: transfer text into numbers
    token = Tokenizer(vocab_size)
    token.fit_on_texts(X_orig)
    X_train_seq = token.texts_to_sequences(X_orig)

    # padding each sample into the same length
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)

    # random shuffle the training set
    '''np.random.seed = 42
    random_index = np.random.permutation(len(X_train_pad))
    X_train = X_train_pad[random_index]
    Y_train = Y_orig[random_index]'''


    # 2. Train your network
    # 		Make sure to print your training loss and accuracy within training to show progress
    # 		Make sure you print the final training accuracy

    # construct NLP model
    word_vector_dim = 32
    NLPmodel = models.Sequential()
    NLPmodel.add(Embedding(vocab_size, word_vector_dim, input_length=max_len))
    NLPmodel.add(Flatten())
    NLPmodel.add(Dense(256, activation='relu'))
    NLPmodel.add(Dense(1, activation='sigmoid'))
    NLPmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = NLPmodel.fit(X_train_pad, Y_orig, epochs=4, batch_size=512, shuffle=True)
    acc = history.history['accuracy']
    print("Final training accuracy is ", acc[3], " after 4 epochs.")


    # 3. Save your model
    NLPmodel.save('20862738_NLP_model.h5')