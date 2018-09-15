# Load LSTM network and generate text
import argparse
import io

parser = argparse.ArgumentParser()
parser.add_argument("text_file", type=str, help="The file containing the original text")
parser.add_argument("model_weights_file", type=str, help="The file containing the trained model weights")
parser.add_argument("model_output_file", type=str, help="The output file where to store the trained model")
parser.add_argument("-n", "--n-hidden", type=str, default="256", help="Number of hidden units per layer, as a comma-separated list")
parser.add_argument("-d", "--dropout", type=str, default="0.2", help="Dropout per layer, as a comma-separated list")
parser.add_argument("-em", "--embedding-length", type=int, default=0, help="Size of vector to use for first layer embedding (if 0 : don't use embedding)")
parser.add_argument("-s", "--sequence-length", type=int, default=100, help="Sequence length")

args = parser.parse_args()

import sys
import numpy
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


def create_model():
	global args, X, n_vocab, seq_length, n_layers

	# define the LSTM model
	model = Sequential()

	n_hidden = args.n_hidden.split(',')
	dropout = args.dropout.split(',')
	n_layers = len(n_hidden)
	if (n_layers != len(dropout)):
		sys.exit("Length of --n-hidden and --dropout do not match.")

	if args.embedding_length <= 0:
		# add first LSTM layer
		model.add(LSTM(int(n_hidden[0]), input_shape=(X.shape[1], X.shape[2]), return_sequences=(n_layers > 1)))
	else:
		# add embedded layer + LSTM layer
		model.add(Embedding(n_vocab, args.embedding_length, input_length=seq_length))
		model.add(LSTM(int(n_hidden[0]), return_sequences=(n_layers > 1)))

	model.add(Dropout(float(dropout[0])))
	for l in range(1, n_layers):
	  model.add(LSTM(int(n_hidden[l]), return_sequences=(l < n_layers-1)))
	  model.add(Dropout(float(dropout[l])))
	model.add(Dense(y.shape[1], activation='softmax'))
	return model

def load_model(model, model_file):
	# load the network weights
	model.load_weights(model_file)
	model.compile(loss='categorical_crossentropy', optimizer='adam')

# load text and covert to lowercase
with io.open(args.text_file, "r", encoding="utf-8") as f:
	raw_text = f.read()
raw_text = raw_text.lower()
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

adataX = numpy.array(dataX)
dataY = numpy.array(dataY)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

if args.embedding_length <= 0:
	# reshape X to be [samples, time steps, features] and normalize
	X = numpy.reshape(dataX, (n_patterns, seq_length, 1)) / float(n_vocab)
else:
	# reshape X to be [samples, time steps]
	X = numpy.reshape(dataX, (n_patterns, seq_length))

model = create_model()
print(model.summary)
load_model(model, args.model_weights_file)
print(model.summary())

print("Saving model")
model.save(args.model_output_file)
