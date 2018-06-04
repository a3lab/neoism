# Load LSTM network and generate text
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("text_file", type=str, help="The file containing the original text")
parser.add_argument("model_file", type=str, help="A file containing the model")
parser.add_argument("output_file", type=str, help="The output file (CSV)")
parser.add_argument("-n", "--n-hidden", type=int, default=256, help="Number of hidden units per layer")
parser.add_argument("-l", "--n-layers", type=int, default=1, help="Number of layers")
parser.add_argument("-s", "--sequence-length", type=int, default=100, help="Sequence length")
parser.add_argument("-e", "--n-epochs", type=int, default=None, help="Number of epochs")
parser.add_argument("-em", "--embedding-length", type=int, default=5, help="Size of vector to use for first layer embedding")
#parser.add_argument("-D", "--model-directory", type=str, default=".", help="The directory where models were saved")
#parser.add_argument("-S", "--sampling_mode", type=str, default="argmax", choices=["argmax", "softmax"], help="Sampling policy")
#parser.add_argument("-N", "--n-words", type=int, default=1000, help="Number of words to generate per epoch/chapter")
#parser.add_argument("-T", "--temperature", type=float, default=1, help="Temperature argument [0, +inf] (for softmax sampling) (higher: more uniform, lower: more greedy")
#parser.add_argument("-E", "--temperature-end", type=float, default=-1, help="Temperature end value (if <= 0 : don't change)")
#parser.add_argument("-b", "--n-best", type=int, default=0, help="Number of best choices from which to pick (to avoid too unlikely outcomes)")
#parser.add_argument("-t", "--transition-factor", type=float, default=0, help="Portion of words in each step that will smoothly transition from one model to the next (in [0, 1])")

args = parser.parse_args()

import sys
import numpy
import os.path
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


def create_model():
	global args, X, n_vocab, seq_length

	# define the LSTM model
	model = Sequential()

	# add embedded layer + LSTM layer
	model.add(Embedding(n_vocab, args.embedding_length, input_length=seq_length))
	model.add(LSTM(args.n_hidden, return_sequences=(args.n_layers > 1)))

	model.add(Dropout(0.2))
	for l in range(1, args.n_layers):
	  model.add(LSTM(args.n_hidden, return_sequences=(l < args.n_layers-1)))
	  model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='softmax'))
	return model

def load_model(model, model_file):
	# load the network weights
	model.load_weights(model_file)
	model.compile(loss='categorical_crossentropy', optimizer='adam')

# load ascii text and covert to lowercase
raw_text = open(args.text_file).read()
raw_text = raw_text.lower()

n_epochs = args.n_epochs

# model files
model_file = args.model_file

# output file
output_file = open(args.output_file, "w")

# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# summarize the loaded data
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


dataX = numpy.array(dataX)
dataY = numpy.array(dataY)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# reshape X to be [samples, time steps]
X = numpy.reshape(dataX, (n_patterns, seq_length))

# Create model
model = create_model()
model.load_weights(model_file)
model.compile(loss='categorical_crossentropy', optimizer='adam')

print(model.summary())

# Get embeddings
weights = model.layers[0].get_weights()[0]

print(type(weights))

# Write to CSV
csv_writer = csv.writer(output_file)
for i in range(len(weights)):
    w = weights[i].tolist()
    c = int_to_char[i]
    csv_writer.writerow( [c] + w )

