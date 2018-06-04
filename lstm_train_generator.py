# -*- coding: utf-8 -*-
# Source: http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("text_file", type=str, help="The file containing the original text")
parser.add_argument("-n", "--n-hidden", type=int, default=256, help="Number of hidden units per layer")
parser.add_argument("-l", "--n-layers", type=int, default=1, help="Number of layers")
parser.add_argument("-s", "--sequence-length", type=int, default=100, help="Sequence length")
parser.add_argument("-m", "--model-file", type=str, default="", help="Model file to load (in order to restart from a certain point)")
parser.add_argument("-i", "--initial-epoch", type=int, default=0, help="Epoch at which to start training (useful for resuming previous training)")
parser.add_argument("-e", "--n-epochs", type=int, default=20, help="Number of epochs to train (total)")
parser.add_argument("-em", "--embedding-length", type=int, default=0, help="Size of vector to use for first layer embedding (if 0 : don't use embedding)")
#parser.add_argument("-lr", "--learning-rate", type=float, default=0.001, help="The learning rate")
parser.add_argument("-D", "--model-directory", type=str, default=".", help="The directory where to save models")
parser.add_argument("-P", "--prefix", type=str, default="lstm-weights-", help="Prefix to use for saving files")
parser.add_argument("-b", "--batch-size", type=int, default=128, help="The batch size")
parser.add_argument("-p", "--batch-save-period", type=int, default=0, help="Period at which to save weights (ie. after every X batch, 0 = no batch save)")

parser.add_argument("-fp", "--first-epoch-params", type=str, default=None, help="A formatted string describing the evolution of batch size and save period during the first epoch")

args = parser.parse_args()

import numpy
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.callbacks import Callback
from keras.utils import np_utils

import time

class ModelSave(Callback):
	def __init__(self, filepath, mode="epoch", save_weights_only=False, period=1):
		super(ModelSave, self).__init__()
		self.filepath = filepath
		self.save_weights_only = save_weights_only
		self.period = period
		self.steps_since_last_save = 0
		if (mode == "epoch"):
			self.process_epoch = True
		elif (mode == "batch"):
			self.process_epoch = False
		else:
			raise ValueError("Option 'mode' must either be set to 'epoch' or 'batch'.")

	def on_step_end(self, filepath, step, logs={}):
		self.steps_since_last_save += 1
		if self.period > 0 and self.steps_since_last_save >= self.period:
			self.steps_since_last_save = 0
			if self.save_weights_only:
				self.model.save_weights(filepath, overwrite=True)
			else:
				self.model.save(filepath, overwrite=True)

	def on_batch_end(self, batch, logs={}):
		if (not self.process_epoch):
			self.on_step_end(self.filepath.format(**logs), batch, logs)

	def on_epoch_end(self, epoch, logs={}):
		if (self.process_epoch):
			self.on_step_end(self.filepath.format(epoch=epoch, **logs), epoch, logs)

# load ascii text and covert to lowercase
raw_text = open(args.text_file).read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = args.sequence_length
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# define the LSTM model
model = Sequential()
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

dataX = numpy.array(dataX)
dataY = numpy.array(dataY)

if args.embedding_length <= 0:
	# reshape X to be [samples, time steps, features] and normalize
	X = numpy.reshape(dataX, (n_patterns, seq_length, 1)) / float(n_vocab)
	# add first LSTM layer
	model.add(LSTM(args.n_hidden, input_shape=(X.shape[1], X.shape[2]), return_sequences=(args.n_layers > 1)))
else:
	# reshape X to be [samples, time steps]
	X = numpy.reshape(dataX, (n_patterns, seq_length))
	# add embedded layer + LSTM layer
	model.add(Embedding(n_vocab, args.embedding_length, input_length=seq_length))
	model.add(LSTM(args.n_hidden, return_sequences=(args.n_layers > 1)))

model.add(Dropout(0.2))
for l in range(1, args.n_layers):
  model.add(LSTM(args.n_hidden, return_sequences=(l < args.n_layers-1)))
  model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

print(model.summary())

if (args.model_file):
  print("Loading weights from: " + args.model_file)
  model.load_weights(args.model_file)

model.compile(loss='categorical_crossentropy', optimizer='adam')
#model.optimizer.lr.set_value(args.learning_rate) # Change learning rate

# define the checkpoint
if not os.path.exists(args.model_directory):
	os.makedirs(args.model_directory)

filepath_prefix="{dir}/{prefix}-layers{n_layers}-nhu{n_hidden}-".format(dir=args.model_directory,prefix=args.prefix,n_hidden=args.n_hidden, n_layers=args.n_layers)
filepath_epoch=filepath_prefix+"e{epoch:03d}.hdf5"
filepath_batch=filepath_prefix+"b{batch:05d}.hdf5"

if (args.initial_epoch == 0):
  model.save_weights(filepath_epoch.format(epoch=-1)) # save startup weights

epoch_model_save = ModelSave(filepath_epoch, mode="epoch", save_weights_only=True)
batch_model_save = ModelSave(filepath_batch, mode="batch", save_weights_only=True, period=args.batch_save_period)

# build first epoch model save
# eg. "10:1:1,20:4:2,100:32:10,-1:128:100"
# save 10 with batch size 1 every 1
# then save 20 with batch size 4 every 2
# then save 100 with batch size 32 every 10
# then save the rest with batch size 128 every 100
def parse_params(params):
	first_epoch_progression = []
	first_epoch_args = params.split(',')
	for arg in first_epoch_args:
		first_epoch_progression.append( [ int(x) for x in arg.split(':') ] )
	return first_epoch_progression

# Parse first period progression specifics.
if args.first_epoch_params != None:
	first_epoch_progression = parse_params(args.first_epoch_params)
else:
	first_epoch_progression = None

n_epochs = args.n_epochs
n_epochs_remaining = n_epochs

# train
absolute_time = time.time()
cpu_time = time.clock()

# train first epoch separately
if args.initial_epoch == 0:
	if first_epoch_progression == None:
		model.fit(X, y, nb_epoch=1, batch_size=args.batch_size, callbacks=[epoch_model_save, batch_model_save])
	else:
		print("Special training for first epoch")
		# simulate sub-epochs
		i = 0
		first_epoch_completed = False
		for k in range(len(first_epoch_progression)):
			param = first_epoch_progression[k]
			n_saves           = param[0]
			batch_size        = param[1]
			batch_save_period = param[2]
			print("Step #{step:02d} n_saves={n_saves} batch_size={batch} batch_save_period={period}".format(step=k, n_saves=n_saves, batch=batch_size, period=batch_save_period))
			if (n_saves == -1):
				# complete training over remaining items
				first_epoch_batch_model_save = ModelSave(filepath_batch, mode="batch", save_weights_only=True, period=batch_save_period)
				first_epoch_completed = True
				model.fit(X[i:], y[i:], nb_epoch=1, batch_size=batch_size, callbacks=[epoch_model_save, first_epoch_batch_model_save])
			else:
				for b in range(n_saves):
					for p in range(batch_save_period):
						j = i + batch_size
						model.train_on_batch(X[i:j], y[i:j])
						i = j
					filepath_batch=filepath_prefix+"bs{size:03d}-s{step:02d}b{batch:05d}.hdf5".format(size=batch_size,step=k,batch=b*batch_save_period	)
					print("Saving to " + filepath_batch)
					model.save_weights(filepath_batch, overwrite=True)

		# complete training over remaining items (unless already completed)
		if not first_epoch_completed:
			model.fit(X[i:], y[i:], nb_epoch=1, batch_size=args.batch_size, callbacks=[epoch_model_save, batch_model_save])
	n_epochs_remaining -= 1

# train other epochs
model.fit(X, y, nb_epoch=n_epochs_remaining, batch_size=args.batch_size, callbacks=[epoch_model_save, batch_model_save], initial_epoch=args.initial_epoch)

absolute_time = time.time()  - absolute_time
cpu_time      = time.clock() - cpu_time
n_epochs = args.n_epochs - args.initial_epoch

print("Total time: {time:.2f} {cputime:.2f}".format(time=absolute_time, cputime=cpu_time))
print("Per epoch:  {time:.2f} {cputime:.2f}".format(time=absolute_time/n_epochs, cputime=cpu_time/n_epochs))
