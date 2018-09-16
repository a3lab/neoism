# Load LSTM network and generate text
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("text_file", type=str, help="The file containing the original text")
parser.add_argument("model_file", type=str, help="The file containing the trained model")
parser.add_argument("-s", "--sequence-length", type=int, default=100, help="Sequence length")
parser.add_argument("-S", "--sampling-mode", type=str, default="argmax", choices=["argmax", "softmax", "special"], help="Sampling policy")
parser.add_argument("-N", "--n-words", type=int, default=1000, help="Number of words to generate")
parser.add_argument("-T", "--temperature", type=float, default=1, help="Temperature argument [0, +inf] (for softmax sampling) (higher: more uniform, lower: more greedy")
parser.add_argument("-b", "--n-best", type=int, default=0, help="Number of best choices from which to pick (to avoid too unlikely outcomes)")

# OSC parameters
parser.add_argument("-I", "--send-ip", default="127.0.0.1", help="The ip of the OSC server")
parser.add_argument("-rP", "--receive-port", type=int, default=5005, help="The port the OSC server is listening on")
parser.add_argument("-sP", "--send-port", type=int, default=5006, help="The port the OSC server is writing on")

args = parser.parse_args()

import sys
import numpy
import tensorflow as tf
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
# This is important to fix error:
# "ValueError: Tensor Tensor("dense_1/Softmax:0", shape=(?, 37), dtype=float32) is not an element of this graph."
# Source of bugfix: https://github.com/keras-team/keras/issues/6462
model._make_predict_function()

saved_weights = model.get_weights()

graph = tf.get_default_graph()

#model.compile(loss='categorical_crossentropy', optimizer='adam')

# model = Sequential()
# model.add(LSTM(args.n_hidden, input_shape=(X.shape[1], X.shape[2]), return_sequences=(args.n_layers > 1)))
# model.add(Dropout(0.2))
# for l in range(1, args.n_layers):
#     model.add(LSTM(args.n_hidden, return_sequences=(l < args.n_layers-1)))
#     model.add(Dropout(0.2))
# model.add(Dense(y.shape[1], activation='softmax'))

print(model.summary())

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
seed = dataX[start]
pattern = seed

print("Seed:")
print(("\"", ''.join([int_to_char[value] for value in pattern]), "\""))

temperature = args.temperature
sampling_mode = args.sampling_mode
n_best = args.n_best

from pythonosc import osc_message_builder, udp_client, dispatcher, osc_server

client = udp_client.SimpleUDPClient(args.send_ip, args.send_port)

def generate_start(unused_addr):
    global has_embeddings, n_best, sampling_mode, model, pattern
    print(args)
    model.reset_states()
    pattern = seed

def generate_next(unused_addr):
    global has_embeddings, n_best, sampling_mode, model, pattern, n_vocab, int_to_char, graph

    if has_embeddings:
        x = numpy.reshape(pattern, (1, len(pattern)))
    else:
        x = numpy.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)

    # This is important to fix error:
    # "ValueError: Tensor Tensor("dense_1/Softmax:0", shape=(?, 37), dtype=float32) is not an element of this graph."
    # Source of bugfix: https://github.com/keras-team/keras/issues/6462
    with graph.as_default():
        # run prediction of model
        prediction = model.predict(x, verbose=0).squeeze()

    # if transition_factor > 0 and i >= transition_start_word:
    #   prediction_next = model_next.predict(x, verbose=0).squeeze()
    #   mix_factor = (n_words - i) / float(transition_n_words)
    #   prediction = mix_factor*prediction + (1-mix_factor)*prediction_next

    # argmax
    if (sampling_mode == "argmax"):
        index = numpy.argmax(prediction)

    # random
    elif (sampling_mode == "softmax"):
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
    if result == '\n':
        result = "        "
    result = result.upper()
    client.send_message("/neoism/text", result)
    pattern = numpy.append(pattern, index)
#        pattern.append(index)
    pattern = pattern[1:len(pattern)]

def set_sampling_mode(unused_addr, v):
    global sampling_mode
    print("Sampling mode: " + str(v))
    sampling_mode = v

def set_temperature(unused_addr, v):
    global temperature
    print("Temp: " + str(v))
    temperature = v

def set_n_best(unused_addr, v):
    global n_best
    print("N. best: " + str(v))
    n_best = int(v)

# Note for offset: order is 0 - input, 1 - forget, 2 - gate, 3 - output
def brain_lstm_cut(unused_addr, group, input, n_inputs, unit, n_units, offset):
    global model
    print("Brain LSTM cell cut {} {} {} {} {} {}".format(group, input, n_inputs, unit, n_units, offset))
    weights = model.get_weights()
    i_f = int(input)
    i_t = int(input+n_inputs)
    for u in range(unit, unit+n_units):
        weights[group][i_f:i_t,4*u+offset] = 0
#    weights[group+1][u_f:u_t] = 0
    model.set_weights(weights)

def brain_lstm_noise(unused_addr, group, input, n_inputs, unit, n_units, offset, noise):
    global model
    print("Brain LSTM cell noise {} {} {} {} {} {} {}".format(group, input, n_inputs, unit, n_units, offset, noise))
    weights = model.get_weights()
    i_f = int(input)
    i_t = int(input+n_inputs)
    for u in range(unit, unit+n_units):
        u_t = 4*u+offset
        weights[group][i_f:i_t,u_t] = saved_weights[group][i_f:i_t,u_t] + numpy.random.normal(0, noise, size=n_inputs)
#    weights[group+1][u_f:u_t] = 0
    model.set_weights(weights)

def brain_lstm_restore(unused_addr, group, input, n_inputs, unit, n_units, offset):
    global model
    print("Brain LSTM cell restore {} {} {} {} {} {}".format(group, input, n_inputs, unit, n_units, offset))
    weights = model.get_weights()
    i_f = int(input)
    i_t = int(input+n_inputs)
    for u in range(unit, unit+n_units):
        u_t = 4*u+offset
        weights[group][i_f:i_t,u_t] = saved_weights[group][i_f:i_t,u_t]
#    weights[group+1][u_f:u_t] = 0
    model.set_weights(weights)

def brain_cut(unused_addr, group, input, n_inputs, unit, n_units):
    global model
    print("Brain cut {} {} {} {} {}".format(group, input, n_inputs, unit, n_units))
    weights = model.get_weights()
    i_f = int(input)
    i_t = int(input+n_inputs)
    u_f = int(unit)
    u_t = int(unit+n_units)
    weights[group][i_f:i_t,u_f:u_t] = 0
#    weights[group+1][u_f:u_t] = 0
    model.set_weights(weights)

def brain_noise(unused_addr, group, input, n_inputs, unit, n_units, noise):
    global model
    print("Brain noise {} {} {} {} {} {}".format(group, input, n_inputs, unit, n_units, noise))
    weights = model.get_weights()
    i_f = int(input)
    i_t = int(input+n_inputs)
    u_f = int(unit)
    u_t = int(unit+n_units)
    weights[group][i_f:i_t,u_f:u_t] = saved_weights[group][i_f:i_t,u_f:u_t] + numpy.random.normal(0, noise, size=(n_inputs, n_units))
#    weights[group+1][u_f:u_t] = 0
    model.set_weights(weights)

# def brain_copy(unused_addr, input, n_inputs, unit, n_units, from_input, from_unit):
#     global model
#     print("Brain copy {} {} {} {}".format(input, n_inputs, unit, n_units, from_input, from_unit))
#     weights = model.layers[1].get_weights()
#     i_f = int(input)
#     i_t = int(input+n_inputs)
#     u_f = int(4*unit)
#     u_t = int(4*(unit+n_units))
#     i_f2 = int(from_input)
#     i_t2 = int(from_input+n_inputs)
#     u_f2 = int(4*from_unit)
#     u_t2 = int(4*(from_unit+n_units))
#     weights[0][i_f:i_t,u_f:u_t] = saved_weights[0][i_f2:i_t2,u_f2:u_t2]
#     weights = model.layers[1].set_weights(weights)
#
def brain_restore(unused_addr, group, input, n_inputs, unit, n_units):
    global model, saved_weights
    print("Brain restore {} {} {} {}".format(input, n_inputs, unit, n_units))
    weights = model.get_weights()
    i_f = int(input)
    i_t = int(input+n_inputs)
    u_f = int(unit)
    u_t = int(unit+n_units)
    weights[group][i_f:i_t,u_f:u_t] = saved_weights[group][i_f:i_t,u_f:u_t]
#    weights[group+1][u_f:u_t] = saved_weights[group+1][u_f:u_t]
    model.set_weights(weights)

# def brain_cut_unit(unused_addr, unit, n_units):
#     global model
#     print("Brain cut unit {}".format(unit))
#     weights = model.layers[1].get_weights()
#     i = int(unit)*4
#     w = weights[0]
#     w[:,i:i+n_units*4] = 0
#     weights[0] = w
#     weights = model.layers[1].set_weights(weights)

dispatcher = dispatcher.Dispatcher()
dispatcher.map("/neoism/start", generate_start)
dispatcher.map("/neoism/next", generate_next)
#dispatcher.map("/neoism/stop", generate_stop, "Stop (and quit)")
dispatcher.map("/neoism/sampling_mode", set_sampling_mode)
dispatcher.map("/neoism/temperature", set_temperature)
dispatcher.map("/neoism/n_best", set_n_best)
dispatcher.map("/neoism/brain_cut", brain_cut)
dispatcher.map("/neoism/brain_noise", brain_noise)
#dispatcher.map("/neoism/brain_copy", brain_copy)
dispatcher.map("/neoism/brain_restore", brain_restore)
dispatcher.map("/neoism/brain_lstm_cut", brain_lstm_cut)
dispatcher.map("/neoism/brain_lstm_noise", brain_lstm_noise)
#dispatcher.map("/neoism/brain_copy", brain_copy)
dispatcher.map("/neoism/brain_lstm_restore", brain_lstm_restore)
# dispatcher.map("/neoism/brain_cut/input", brain_cut_input)
# dispatcher.map("/neoism/brain_cut/unit", brain_cut_unit)

server = osc_server.ThreadingOSCUDPServer( ("127.0.0.1", args.receive_port), dispatcher)
print("Serving on {}".format(server.server_address))
server.serve_forever()


print("\nDone.")
