# -*- coding: utf-8 -*-
# Source: http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

import argparse

import numpy
import os
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.callbacks import Callback
from keras.utils import np_utils

from hyperopt import Trials, STATUS_OK, tpe

from hyperas import optim
from hyperas.distributions import choice, uniform


def data():
	parser = argparse.ArgumentParser()
	parser.add_argument("text_file", type=str, help="The file containing the original text")
	parser.add_argument("-l", "--n-layers", type=int, default=1, help="Number of layers")
	parser.add_argument("-s", "--sequence-length", type=int, default=100, help="Sequence length")
	args = parser.parse_args()
	seq_length = args.sequence_length

	raw_text = open(args.text_file).read()
	raw_text = raw_text.lower()

	chars = sorted(list(set(raw_text)))
	char_to_int = dict((c, i) for i, c in enumerate(chars))

	n_chars = len(raw_text)
	n_vocab = len(chars)
	print("Total Characters: ", n_chars)
	print("Total Vocab: ", n_vocab)

	dataX = []
	dataY = []
	for i in range(0, n_chars - seq_length, 1):
		seq_in = raw_text[i:i + seq_length]
		seq_out = raw_text[i + seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])

	n_patterns = len(dataX)
	nTrain = int(n_patterns * 0.8)

	dataX = numpy.array(dataX)
	dataY = numpy.array(dataY)

	print("Total Patterns: ", n_patterns)

	y = np_utils.to_categorical(dataY)
	X = numpy.reshape(dataX, (n_patterns, seq_length))

	xTrain, xTest = numpy.split(X, [nTrain])
	yTrain, yTest = numpy.split(y, [nTrain])
	return xTrain, yTrain, xTest, yTest

def create_model(xTrain, yTrain, xTest, yTest):
	model = Sequential()

	model.add(Embedding(n_vocab, output_dim={{choice([3,4,5,7,9])}}, input_length=seq_length))
	model.add(LSTM(units={{choice([64,128,256,512,1024])}}, return_sequences=True))
	model.add(Dropout({{uniform(0,1)}}))

	model.add(LSTM(units={{choice([64,128,256,512,1024])}}, return_sequences=False))
	model.add(Dropout({{uniform(0,1)}}))
	
	model.add(Dense(y.shape[1], activation='softmax'))

	print(model.summary())

	model.compile(loss='categorical_crossentropy', optimizer={{choice(['rmsprop', 'adam', 'sgd'])}}, metrics=["accuracy"])

	print("Fitting...")
	model.fit(xTrain, yTrain, batch_size={{choice([32,64,128,256,512,1024])}},epochs=1,verbose=2,validation_data=(xTest,yTest))

	print("Evaluating...")
	score, acc = model.evaluate(xTest, yTest, verbose=0)

	print('Test accuracy:', acc)
	return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':


	best_run, best_model = optim.minimize(model=create_model,
		                              data=data,
		                              algo=tpe.suggest,
		                              max_evals=10,verbose=2,
		                              trials=Trials())
	print("Done optimizing")
	xTrain, yTrain, xTest, yTest = data()
	print("Evalutation of best performing model:")
	print(best_model.evaluate(xTest, yTest))
	print("Best performing model chosen hyper-parameters:")
	print(best_run)


