# Load LSTM network and generate text
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("text_file", type=str, help="The file containing the original text")
parser.add_argument("model_file", type=str, help="The file containing the trained model")
parser.add_argument("-n", "--n-hidden", type=int, default=256, help="Number of hidden units per layer")
parser.add_argument("-l", "--n-layers", type=int, default=1, help="Number of layers")
parser.add_argument("-s", "--sequence-length", type=int, default=100, help="Sequence length")
parser.add_argument("-em", "--embedding-length", type=int, default=0, help="Size of vector to use for first layer embedding (if 0 : don't use embedding)")
parser.add_argument("-S", "--sampling-mode", type=str, default="argmax", choices=["argmax", "softmax", "special"], help="Sampling policy")
parser.add_argument("-N", "--n-words", type=int, default=1000, help="Number of words to generate")
parser.add_argument("-T", "--temperature", type=float, default=1, help="Temperature argument [0, +inf] (for softmax sampling) (higher: more uniform, lower: more greedy")

args = parser.parse_args()

import sys
import numpy
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# load ascii text and covert to lowercase
raw_text = open(args.text_file).read()
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


dataX = numpy.array(dataX)
dataY = numpy.array(dataY)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

n_words = args.n_words

# define the LSTM model
model = load_model(args.model_file)

# model = Sequential()
# model.add(LSTM(args.n_hidden, input_shape=(X.shape[1], X.shape[2]), return_sequences=(args.n_layers > 1)))
# model.add(Dropout(0.2))
# for l in range(1, args.n_layers):
# 	model.add(LSTM(args.n_hidden, return_sequences=(l < args.n_layers-1)))
# 	model.add(Dropout(0.2))
# model.add(Dense(y.shape[1], activation='softmax'))

print((model.summary()))

has_embeddings = (isinstance(model.layers[0], Embedding))

if has_embeddings:
	# reshape X to be [samples, time steps]
	X = numpy.reshape(dataX, (n_patterns, seq_length))
else:
	# reshape X to be [samples, time steps, features] and normalize
	X = numpy.reshape(dataX, (n_patterns, seq_length, 1)) / float(n_vocab)


# # load the network weights
# model.load_weights(args.model_file)
# model.compile(loss='categorical_crossentropy', optimizer='adam')

# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]

print("Seed:")
print(("\"", ''.join([int_to_char[value] for value in pattern]), "\""))

temperature = args.temperature

# generate characters
for i in range(n_words):
	if has_embeddings:
		x = numpy.reshape(pattern, (1, len(pattern)))
	else:
		x = numpy.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)

	# run prediction of model
	prediction = model.predict(x, verbose=0).squeeze()

	# if transition_factor > 0 and i >= transition_start_word:
	#   prediction_next = model_next.predict(x, verbose=0).squeeze()
	#   mix_factor = (n_words - i) / float(transition_n_words)
	#   prediction = mix_factor*prediction + (1-mix_factor)*prediction_next

	# argmax
	if (args.sampling_mode == "argmax"):
		index = numpy.argmax(prediction)

	# random
	elif (args.sampling_mode == "softmax"):
		# Source: https://gist.github.com/alceufc/f3fd0cd7d9efb120195c
		if (temperature != 1):
			prediction = numpy.power(prediction, 1./temperature)
			prediction /= prediction.sum(0)

		#print "[" + args.sampling_mode +"]"
		#print (args.sampling_mode == "softmax")

		if n_best == 0:
			#print "using softmax"
			index = numpy.asscalar(numpy.random.choice(numpy.arange(n_vocab), 1, p=prediction))

		# special
		else:
			prediction = numpy.asarray(prediction)
			max_indices = prediction.argsort()[-n_best:][::-1]
			max_indices_weights = [ prediction[m] for m in max_indices ]
			max_indices_sum = numpy.sum(max_indices_weights)
			max_indices_weights = [ prediction[m]/max_indices_sum for m in max_indices ]
			index = numpy.asscalar(numpy.random.choice(numpy.array(max_indices), 1, p=numpy.array(max_indices_weights)))

	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern = numpy.append(pattern, index)
#		pattern.append(index)
	pattern = pattern[1:len(pattern)]
print("\nDone.")
