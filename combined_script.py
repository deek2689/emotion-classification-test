#!/usr/bin/env python3

import numpy as np
import os
import sys

import wave
import copy
import math


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM, Input, Flatten, Embedding, Convolution1D,Dropout
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model




from features import *
from helper import *


code_path = os.path.dirname(os.path.realpath(os.getcwd()))
emotions_used = np.array(['ang', 'exc', 'neu', 'sad'])
#data_path = code_path + "/../data/sessions/"
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
framerate = 16000


import pickle
import pickle5 as p
#with open(data_path + '/../'+'data_collected.pickle'/home/sentrybot/Downloads, 'rb') as handle:
   # data2 = pickle.load(handle)
with open('/home/sentrybot/Downloads/data_collected_nocap.pickle', 'rb') as handle:
    data2 = p.load(handle,encoding = 'bytes')    

text = []

for ses_mod in data2:
    text.append(ses_mod['transcription'])
    
MAX_SEQUENCE_LENGTH = 500

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

token_tr_X = tokenizer.texts_to_sequences(text)
x_train_text = []

x_train_text = sequence.pad_sequences(token_tr_X, maxlen=MAX_SEQUENCE_LENGTH)

import codecs
EMBEDDING_DIM = 300

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

file_loc = '/home/sentrybot/Downloads/crawl-300d-2M.vec'

print (file_loc)

gembeddings_index = {}
with codecs.open(file_loc, encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        gembedding = np.asarray(values[1:], dtype='float32')
        gembeddings_index[word] = gembedding
#
f.close()
print('G Word embeddings:', len(gembeddings_index))

nb_words = len(word_index) +1
g_word_embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    gembedding_vector = gembeddings_index.get(word)
    if gembedding_vector is not None:
        g_word_embedding_matrix[i] = gembedding_vector
        
print('G Null word embeddings: %d' % np.sum(np.sum(g_word_embedding_matrix, axis=1) == 0))


def calculate_features(frames, freq, options):
    window_sec = 0.2
    window_n = int(freq * window_sec)

    st_f = stFeatureExtraction(frames, freq, window_n, window_n / 2)

    if st_f.shape[1] > 2:
        i0 = 1
        i1 = st_f.shape[1] - 1
        if i1 - i0 < 1:
            i1 = i0 + 1
        
        deriv_st_f = np.zeros((st_f.shape[0], i1 - i0), dtype=float)
        for i in range(i0, i1):
            i_left = i - 1
            i_right = i + 1
            deriv_st_f[:st_f.shape[0], i - i0] = st_f[:, i]
        return deriv_st_f
    elif st_f.shape[1] == 2:
        deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
        deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
        return deriv_st_f
    else:
        deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
        deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
        return deriv_st_f
        
x_train_speech = []

counter = 0
for ses_mod in data2:
    x_head = ses_mod['signal']
    st_features = calculate_features(x_head, framerate, None)
    st_features, _ = pad_sequence_into_array(st_features, maxlen=100)
    x_train_speech.append( st_features.T )
    counter+=1
    if(counter%100==0):
        print(counter)
    
x_train_speech = np.array(x_train_speech)
print(x_train_speech.shape)

#Y=[]
#for ses_mod in data2:
    #Y.append(ses_mod['emotion'])
    

            
Y=[e['emotion'] for e in data2 if e['emotion'] in emotions_used]    
Y = label_binarize(Y, emotions_used)

#Y.shape

    
#Y = label_binarize(Y,emotions_used)

print(Y.shape)


model_text = Sequential()
#model.add(Embedding(2737, 128, input_length=MAX_SEQUENCE_LENGTH))
model_text.add(Embedding(2736,
                    128,input_length=500))

model_text.add(LSTM(256, return_sequences=True, input_shape=(100, 34)))
model_text.add(LSTM(256, return_sequences=False))
model_text.add(Dense(256))


model_speech = Sequential()
model_speech.add(Flatten(input_shape=(100, 34)))
model_speech.add(Dense(1024))
model_speech.add(Activation('relu'))
model_speech.add(Dropout(0.2))
model_speech.add(Dense(256))


model_combined = Sequential()
model_combined.add(Concatenate([model_text, model_speech]))
model_combined.add(Activation('relu'))

model_combined.add(Dense(256))
model_combined.add(Activation('relu'))

model_combined.add(Dense(4))
model_combined.add(Activation('softmax'))
#out = Dense(4, activation='softmax', name='output')(concatenated_models)

#added_model = Model([model_text, model_speech], out)

#added_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#added_model.fit([x_train_text,x_train_speech], Y, epochs=125, batch_size=64, verbose=1)

#inputs = Input(shape=(100,34),name='Input_1')
#output1 = Dense(4, name='Dense_1')(model_combined)
#model_C = Model(inputs=inputs, outputs=output1)
#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model_combined.compile(loss='categorical_crossentropy',optimizer='adam' ,metrics=['acc'])
model_combined.build(([34, 34], 4))
#model_combined.build(input_shape=(100,34))
## compille it here according to instructions

#model.compile()
print(model_speech.summary())
print(model_text.summary())
#model_combined.build(input_shape=(1,34,34,1))
print(model_combined.summary())

#print("Model1 Built")




#hist = model_combined.fit([x_train_text,x_train_speech], Y, epochs=125, verbose=1, validation_split=0.2)

#print(hist)

#print(added_model.summary())

from tensorflow.keras.utils import plot_model

plot_model(model_combined)

model.save('/home/sentrybot/Downloads/models/combined.h5')
