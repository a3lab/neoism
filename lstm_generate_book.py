# Load LSTM network and generate text
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("text_file", type=str, help="The file containing the original text")
parser.add_argument("model_file_list", type=str, help="A file containing all the model files to use OR a prefix to use for files")
parser.add_argument("output_file", type=str, help="The output file")
parser.add_argument("-n", "--n-hidden", type=int, default=256, help="Number of hidden units per layer")
parser.add_argument("-l", "--n-layers", type=int, default=1, help="Number of layers")
parser.add_argument("-s", "--sequence-length", type=int, default=100, help="Sequence length")
parser.add_argument("-e", "--n-epochs", type=int, default=None, help="Number of epochs")
parser.add_argument("-em", "--embedding-length", type=int, default=0, help="Size of vector to use for first layer embedding (if 0 : don't use embedding)")
parser.add_argument("-D", "--model-directory", type=str, default=".", help="The directory where models were saved")
parser.add_argument("-S", "--sampling_mode", type=str, default="argmax", choices=["argmax", "softmax"], help="Sampling policy")
parser.add_argument("-N", "--n-words", type=int, default=1000, help="Number of words to generate per epoch/chapter")
parser.add_argument("-T", "--temperature", type=float, default=1, help="Temperature argument [0, +inf] (for softmax sampling) (higher: more uniform, lower: more greedy")
parser.add_argument("-E", "--temperature-end", type=float, default=-1, help="Temperature end value (if <= 0 : don't change)")
parser.add_argument("-b", "--n-best", type=int, default=0, help="Number of best choices from which to pick (to avoid too unlikely outcomes)")
parser.add_argument("-t", "--transition-factor", type=float, default=0, help="Portion of words in each step that will smoothly transition from one model to the next (in [0, 1])")
parser.add_argument("-p", "--seed", type=str, default=None, help="The seed used to generate the text (default will use end of training text)")

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

	if args.embedding_length <= 0:
		# add first LSTM layer
		model.add(LSTM(args.n_hidden, input_shape=(X.shape[1], X.shape[2]), return_sequences=(args.n_layers > 1)))
	else:
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
if os.path.isfile(args.model_file_list):
	model_files = [ line.rstrip('\n').strip() for line in open(args.model_file_list)]
else:
	model_files = [	"{prefix}{epoch:02d}.hdf5".format(prefix=args.model_file_list, epoch=e) for e in range(n_epochs) ]
model_files = [ args.model_directory + "/" + f for f in model_files ]
if n_epochs == None or n_epochs > len(model_files):
	n_epochs = len(model_files)

# output file
output_file = open(args.output_file, "w+")

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

if args.embedding_length <= 0:
	# reshape X to be [samples, time steps, features] and normalize
	X = numpy.reshape(dataX, (n_patterns, seq_length, 1)) / float(n_vocab)
else:
	# reshape X to be [samples, time steps]
	X = numpy.reshape(dataX, (n_patterns, seq_length))

n_words = args.n_words
transition_factor = args.transition_factor
transition_n_words = int(n_words * transition_factor)
transition_start_word = n_words - transition_n_words

# Create model
model = create_model()
if transition_factor > 0:
  model_next = create_model()
else:
  model_next = None

print(model.summary())

# pick end of text as seed
if (args.seed == None):
	pattern = dataX[-1]
# of use predefined seed
else:
	pattern = [char_to_int[char] for char in args.seed[-seq_length:]]

print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

temperature = args.temperature

n_best = args.n_best

# Run through all epochs, outputing text
for e in range(n_epochs):
	if (args.temperature_end > 0):
		# linear interpolation
		temperature = args.temperature + (float(e)/(n_epochs-1)) * (args.temperature_end - args.temperature)

	model_file = model_files[e]
	print("Generating epoch/step # {epoch} (temperature={temp}) using file {filename}".format(epoch=e,temp=temperature,filename=model_file))
	# load the network weights
	model.load_weights(model_file)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	#output_file.write("\n\nEPOCH {epoch}\n\n".format(epoch=e))

	if transition_factor > 0 and e < n_epochs-1:
		model_next.load_weights(model_files[e+1])
		model_next.compile(loss='categorical_crossentropy', optimizer='adam')

	# generate characters
	for i in range(n_words):
		if args.embedding_length <= 0:
			x = numpy.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
		else:
			x = numpy.reshape(pattern, (1, len(pattern)))

		# run prediction of model
		prediction = model.predict(x, verbose=0).squeeze()

		if transition_factor > 0 and i >= transition_start_word:
		  prediction_next = model_next.predict(x, verbose=0).squeeze()
		  mix_factor = (n_words - i) / float(transition_n_words)
		  prediction = mix_factor*prediction + (1-mix_factor)*prediction_next

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
		output_file.write(result)
		pattern = numpy.append(pattern, index)
#		pattern.append(index)
		pattern = pattern[1:len(pattern)]

	print("Done")
	output_file.flush()
