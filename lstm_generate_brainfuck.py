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

parser.add_argument("-r", "--frame-rate", type=float, default=10, help="Number of letters per second")

# OSC parameters
parser.add_argument("-I", "--send-ip", default="127.0.0.1", help="The ip of the OSC server")
parser.add_argument("-rP", "--receive-port", type=int, default=5005, help="The port the OSC server is listening on")
parser.add_argument("-sP", "--send-port", type=int, default=5006, help="The port the OSC server is writing on")

parser.add_argument("--no-arduino", action='store_true', default=False, help="Optional flag to run the script without checking the Arduino - for testing purposes")
parser.add_argument("--arduino-serial-number", default=None, help="The serial number of the Arduino (to auto-find the serial port)")
parser.add_argument("--baudrate", default=115200, help="The baudrate")

args = parser.parse_args()

from pythonosc import osc_message_builder, udp_client, dispatcher, osc_server
import threading
import colorsys
client = udp_client.SimpleUDPClient(args.send_ip, args.send_port)

import serial
import serial.tools.list_ports

def find_arduino(serial_number):
    for pinfo in serial.tools.list_ports.comports():
        if pinfo.serial_number == serial_number:
            return serial.Serial(pinfo.device,args.baudrate,timeout=5)
    raise IOError("Could not find an arduino - is it plugged in?")

def new_seed():
    global dataX, char_to_int, int_to_char, seq_length

    # pick a random seed
    start = numpy.random.randint(0, len(dataX) - 1)

    # move until you find a '.'
    point = char_to_int['.']

    while dataX[start][seq_length-1] != point:
        start = (start + 1) % len(dataX)

    # Somehow we need to make sure to add one extra space, otherwise the LSTM somehow fails to generate a space after a dot.
    start = (start + 1) % len(dataX)
    seed = dataX[start]

    print("New seed:")
    print(("\"", ''.join([int_to_char[value] for value in seed]), "\""))
    return seed

enable_arduino = not args.no_arduino

if enable_arduino:
    ard = find_arduino(serial_number=args.arduino_serial_number)

import time
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

pattern = new_seed()

temperature = args.temperature
sampling_mode = args.sampling_mode
n_best = args.n_best


# This class represents the current state of the machine in terms of its connections.
class State:

    N_PINS = 64
    PORTS = [
        "31!", "32!", "33!",

        "31?", "32?", "33?",               "34?", "35?", "36?", "37?",
        "34!", "35!", "36!", "37!",        "21!", "22!", "23!", "24!", "25!", "26!", "27!", "28!",

        "21?", "22?", "23?", "24?", "25?", "26?", "27?", "28?",        "29?", "2A?", "2B?", "2C?",
        "29!", "2A!", "2B!", "2C!",        "11!", "12!", "13!", "14!", "15!", "16!", "17!", "18!",

        "11?", "12?", "13?", "14?", "15?", "16?", "17?", "18?",

        "01!", "02!", "03!", "04!", "05!",
        "01?", "02?", "03?", "04?", "05?"
    ]
    connections = None

    def __init__(self):
        # Create array of arrays.
        self.connections = [ [ False for j in range(self.N_PINS) ] for i in range(self.N_PINS) ]

    def copy_from(self, other):
        self.connections = other.connections

    def reset(self):
        for i in range(self.N_PINS):
            for j in range(self.N_PINS):
                self.disconnect(i, j)

    def connect(self, fromId, toId):
        self.set_connect(fromId, toId, True)

    def disconnect(self, fromId, toId):
        self.set_connect(fromId, toId, False)

    def set_connect(self, fromId, toId, state):
        self.connections[fromId][toId] = state
        self.connections[toId][fromId] = state

    def is_connected(self, fromId, toId):
        return self.connections[fromId][toId]

    def get_port_id(self, portname):
        return self.PORTS.index(portname)

    def get_port_name(self, portId):
        return self.PORTS[portId]

    def get_all_ports(self):
        return self.PORTS

    def get_disconnected_ports(self):
        disconnected = []
        for port in self.PORTS:
            conn = self.get_connections(port)
            if not conn:
                disconnected.append(port)
        return disconnected

    def get_misconnected_ports(self):
        misconnected = []
        for port in self.PORTS:
            conn = self.get_connections(port)
            if conn:
                port_info = self.get_info(port)
                info = self.get_info(conn[0])
                if info["code"] != port_info["code"]:
                    misconnected.append(port)
        return misconnected

    def get_connections(self, portname):
        conn = []
        portId = self.get_port_id(portname)
        for i in range(self.N_PINS):
            if i != portId and self.is_connected(portId, i):
                conn.append(self.get_port_name(i))
        return conn

    def get_info(self, portname):
        info = {
            "name": portname,
            "code" : portname[0:2],
            "level": int(portname[0]),
            "position": portname[1],
            "input": portname[2] == "!"
        }
        return info

    def debug(self):
        for p in self.PORTS:
            c = self.get_connections(p)
            if c:
                print("{} => {}".format(p, c))

def lerp(value, fromLow, fromHigh, toLow, toHigh):
    # Avoid divisions by zero.
    if fromLow == fromHigh:
        return (toLow + toHigh) / 2.0 # dummy value
    return (value - fromLow) * (toHigh - toLow) / (fromHigh - fromLow) + toLow

def process_group(connection_info, weights, port_names, group, block_size, n_units, noise_level, lstm=False):
    global state

#    print("processing group: {}".format(port_names))

    from_i = 0
    lstm_offset = 0

    dis_connection_count = 0
    mis_connection_count = 0
    mis_connection_bypass = 0
    for port in port_names:
        connections = state.get_connections(port)
        if not connections:
            dis_connection_count += 1
        else:
            port_info = state.get_info(port)
            info = state.get_info(connections[0])
            if info["code"] != port_info["code"]: # mis-connection: not the "right" port
                mis_connection_count += 1
                mis_connection_bypass += abs(info["level"] - port_info["level"])

    connection_quality = 1.0 - ((mis_connection_count * 0.5 + dis_connection_count) / len(port_names))
    mis_connection_level = mis_connection_count / len(port_names)

    for port in port_names:
        connections = state.get_connections(port)
        arguments = [ weights, group, from_i, block_size, 0, n_units]
        if lstm:
            arguments += [lstm_offset]

        if not connections:
            print("Port {} disconnected".format(port))
            arguments += [ connection_quality ]
            if lstm:
                weights_lstm_cut(*arguments)
            else:
                weights_cut(*arguments)
        else:
            port_info = state.get_info(port)
            info = state.get_info(connections[0])
            if info["code"] == port_info["code"]: # correct port
                pass
                # if lstm:
                #     weights_lstm_restore(*arguments)
                # else:
                #     weights_restore(*arguments)
            else:
                print("Port {} misconnected".format(port))
                arguments += [ mis_connection_level * noise_level ]
                if lstm:
                    weights_lstm_noise(*arguments)
                else:
                    weights_noise(*arguments)
#                speed_adjust += port_info["level"] - info["level"]

        # Special case: for LSTM layers only update the from_i variable at the end of offset cycles
        if lstm:
            lstm_offset += 1
            if lstm_offset >= 4:
                lstm_offset = 0
                from_i += block_size
        else:
            from_i += block_size

    # Update connection info.
    connection_info[0] += dis_connection_count
    connection_info[1] += mis_connection_count
    connection_info[2] += mis_connection_bypass

import copy
def process_state():
    global model, temperature, n_best

    # Reload old weights.
    weights = copy.deepcopy(saved_weights)

#    print("processsing state")
    n_embeddings = 5
    n_characters = 56
    n_hidden_layer_1 = 64
    n_hidden_layer_2 = 1024

    # Computes the toal : (disconnections, misconnections, bypasses)
    connection_info = [0, 0, 0]

    # Output layer.
#    print("Process output")
    process_group(connection_info, weights, [ "31!", "32!", "33!" ], 7, int(n_hidden_layer_2 / 3), n_characters, 0.1)

    # LSTM layer #2
#    print("Process layer 2")
    process_group(connection_info, weights, [ "21!", "22!", "23!", "24!", "25!", "26!", "27!", "28!" ], 4, int(n_hidden_layer_1 / 2), n_hidden_layer_2, 0.1, True)
    process_group(connection_info, weights, [ "34!", "35!", "36!", "37!" ], 5, n_hidden_layer_2, n_hidden_layer_2, 0.05, True)

    # LSTM layer #1
#    print("Process layer 1")
    process_group(connection_info, weights, [ "11!", "12!", "13!", "14!", "15!", "16!", "17!", "18!" ], 1, int(n_embeddings / 2), n_hidden_layer_1, 0.1, True)
    process_group(connection_info, weights, [ "29!", "2A!", "2B!", "2C!" ], 2, n_hidden_layer_1, n_hidden_layer_1, 0.05, True)

    # Embeddings layer
#    print("Process embeddings")
    process_group(connection_info, weights, [ "01!", "02!", "03!", "04!", "05!" ], 0, int(n_characters / 5), n_embeddings, 0.1)

    # Implement changes.
    model.set_weights(weights)

    n_possible_connections = 32
    n_possible_bypass = (3*7 + 2*12 + 2*12 + 3*5)
    dis_connection_level = connection_info[0] / n_possible_connections
    mis_connection_level = connection_info[1] / n_possible_connections
    mis_connection_bypass_level = connection_info[2] / n_possible_bypass

    # More activity => higher temperatures
    activity_level = - dis_connection_level + mis_connection_level
    new_temperature = lerp(activity_level, -1, 1, 0.1, 1.1)

    # More bypass = more speed.
    speed_level = - dis_connection_level + mis_connection_bypass_level
    frame_rate      = lerp(speed_level, -1, 1, 2.5, 20)
    set_frame_rate(frame_rate)

    # Strange arrangement:
    if mis_connection_bypass_level > 0:
        n_best = int(lerp(mis_connection_bypass_level, 0, 1, n_characters/2, 1))
    elif mis_connection_level > 0:
        n_best = int(lerp(mis_connection_level, 0, 1, n_characters/2, n_characters))
        if n_best >= n_characters:
            n_best = 0
    else:
        n_best = int(n_characters / 2)

    print("temp={} nb={} dis={}% mis={}% misb={}% act={} ({})"
            .format(temperature, n_best, dis_connection_level*100, mis_connection_level*100, mis_connection_bypass_level*100,
                    activity_level, connection_info))

    # If temperature has changed: update text color.
    if new_temperature != temperature:
        temperature = new_temperature
        update_color(lerp(activity_level, -1, 1, 0, 1))

def set_frame_rate(fps):
    global period
    period = 1 / fps

def generate_start(unused_addr=None):
    global has_embeddings, n_best, sampling_mode, model, pattern
    print(args)
    model.reset_states()
    pattern = new_seed()

def generate_next(unused_addr=None):
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
    result = result.upper()
    if result == '\n':
        result = " *** "
    client.send_message("/neoism/text", result)
    pattern = numpy.append(pattern, index)
#        pattern.append(index)
    pattern = pattern[1:len(pattern)]

def update_color(value):
    global client
    hue = lerp(value, 0, 1, 0.67, -0.336)
    if hue < 0:
        hue += 1
    print("HUE : {} {}".format(value, hue))
    color = colorsys.hsv_to_rgb(hue, 1, 1)
    r = int(color[0] * 255)
    g = int(color[1] * 255)
    b = int(color[2] * 255)
    client.send_message("/neoism/color", [r, g, b])

def set_sampling_mode(unused_addr, v):
    global sampling_mode
    print("Sampling mode: " + str(v))
    sampling_mode = v

def test_misconnection_color(unused_addr, v):
    update_color(v)

def set_temperature(unused_addr, v):
    global temperature
    print("Temp: " + str(v))
    temperature = v

def set_n_best(unused_addr, v):
    global n_best
    print("N. best: " + str(v))
    n_best = int(v)

# Note for offset: order is 0 - input, 1 - forget, 2 - gate, 3 - output
def weights_lstm_cut(weights, group, input, n_inputs, unit, n_units, offset, connection_level):
    print("Brain LSTM cell cut {} {} {} {} {} {}".format(group, input, n_inputs, unit, n_units, offset))
    i_f = int(input)
    i_t = int(input+n_inputs)
    for u in range(unit, unit+n_units):
        u_t = 4*u+offset
        weights[group][i_f:i_t,4*u+offset] = saved_weights[group][i_f:i_t,u_t] * connection_level
#    weights[group+1][u_f:u_t] = 0

def weights_lstm_noise(weights, group, input, n_inputs, unit, n_units, offset, noise):
    global saved_weights
    print("Brain LSTM cell noise {} {} {} {} {} {} {}".format(group, input, n_inputs, unit, n_units, offset, noise))
    i_f = int(input)
    i_t = int(input+n_inputs)
    for u in range(unit, unit+n_units):
        u_t = 4*u+offset
        weights[group][i_f:i_t,u_t] = saved_weights[group][i_f:i_t,u_t] + numpy.random.normal(0, noise, size=n_inputs)
#    weights[group+1][u_f:u_t] = 0

def weights_lstm_restore(weights, group, input, n_inputs, unit, n_units, offset):
    global saved_weights
    print("Brain LSTM cell restore {} {} {} {} {} {}".format(group, input, n_inputs, unit, n_units, offset))
    i_f = int(input)
    i_t = int(input+n_inputs)
    for u in range(unit, unit+n_units):
        u_t = 4*u+offset
        weights[group][i_f:i_t,u_t] = saved_weights[group][i_f:i_t,u_t]
#    weights[group+1][u_f:u_t] = 0

def weights_cut(weights, group, input, n_inputs, unit, n_units, connection_level):
    print("Brain cut {} {} {} {} {}".format(group, input, n_inputs, unit, n_units))
    i_f = int(input)
    i_t = int(input+n_inputs)
    u_f = int(unit)
    u_t = int(unit+n_units)
    weights[group][i_f:i_t,u_f:u_t] = saved_weights[group][i_f:i_t,u_f:u_t] * connection_level
#    weights[group+1][u_f:u_t] = 0

def weights_noise(weights, group, input, n_inputs, unit, n_units, noise):
    global saved_weights
    print("Brain noise {} {} {} {} {} {}".format(group, input, n_inputs, unit, n_units, noise))
    i_f = int(input)
    i_t = int(input+n_inputs)
    u_f = int(unit)
    u_t = int(unit+n_units)
    weights[group][i_f:i_t,u_f:u_t] = saved_weights[group][i_f:i_t,u_f:u_t] + numpy.random.normal(0, noise, size=(n_inputs, n_units))
#    weights[group+1][u_f:u_t] = 0

# Note for offset: order is 0 - input, 1 - forget, 2 - gate, 3 - output
def brain_lstm_cut(unused_addr, group, input, n_inputs, unit, n_units, offset):
    global model
    print("Brain LSTM cell cut {} {} {} {} {} {}".format(group, input, n_inputs, unit, n_units, offset))
    weights = model.get_weights()
    weights_lstm_cut(weights, group, input, n_inputs, unit, n_units, offset, 1.0)
    model.set_weights(weights)

def brain_lstm_noise(unused_addr, group, input, n_inputs, unit, n_units, offset, noise):
    global model
    weights = model.get_weights()
    weights_lstm_noise(weights, group, input, n_inputs, unit, n_units, offset, noise)
    model.set_weights(weights)

def brain_lstm_restore(unused_addr, group, input, n_inputs, unit, n_units, offset):
    global model
    print("Brain LSTM cell restore {} {} {} {} {} {}".format(group, input, n_inputs, unit, n_units, offset))
    weights = model.get_weights()
    weights_lstm_restore(weights, group, input, n_inputs, unit, n_units, offset)
    model.set_weights(weights)

def brain_cut(unused_addr, group, input, n_inputs, unit, n_units):
    global model
    print("Brain cut {} {} {} {} {}".format(group, input, n_inputs, unit, n_units))
    weights = model.get_weights()
    weights_cut(weights, group, input, n_inputs, unit, n_units, 1.0)
    model.set_weights(weights)

def brain_noise(unused_addr, group, input, n_inputs, unit, n_units, noise):
    global model
    print("Brain noise {} {} {} {} {} {}".format(group, input, n_inputs, unit, n_units, noise))
    weights = model.get_weights()
    weights_noise(weights, group, input, n_inputs, unit, n_units, noise)
    model.set_weights(weights)

def brain_restore(unused_addr, group, input, n_inputs, unit, n_units):
    global model, saved_weights
    print("Brain restore {} {} {} {}".format(input, n_inputs, unit, n_units))
    weights = model.get_weights()
    weights_restore(weights, group, input, n_inputs, unit, n_units)
    model.set_weights(weights)

# Procedure ran at switch off.
def send_message(msg, wait=1):
    global client
    client.send_message("/neoism/text", msg) # clear screen
    if wait:
        time.sleep(wait)

def switch_off():
    global client, state
    send_message("  ---  ")
    dis_connected = state.get_disconnected_ports()
    mis_connected = state.get_misconnected_ports()
    if dis_connected:
        send_message(" DISC: ", 2)
        if len(dis_connected) >= 10:
            send_message("TOOMANY")
            send_message("       N. {}".format(len(dis_connected)))
        else:
            send_message("       ", 0.5) # clear screen
            for p in dis_connected:
                send_message(p + " ")
    if mis_connected:
        print(mis_connected)
        send_message(" MISC: ", 2)
        if len(mis_connected) >= 10:
            send_message("TOOMANY")
            send_message("       N. {}".format(len(mis_connected)))
        else:
            send_message("       ", 0.5) # clear screen
            for p in mis_connected:
                send_message(p + " ")
    send_message("       ") # clear screen

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
dispatcher.map("/neoism/test_misconnection_color", test_misconnection_color)
# dispatcher.map("/neoism/brain_cut/input", brain_cut_input)
# dispatcher.map("/neoism/brain_cut/unit", brain_cut_unit)

server = osc_server.ThreadingOSCUDPServer( ("127.0.0.1", args.receive_port), dispatcher)
print("Serving on {}".format(server.server_address))
server_thread = threading.Thread(target=server.serve_forever)
server_thread.start()

state = State()
new_state = State()

period = None
set_frame_rate(args.frame_rate)

generate_start()
start_time = time.time()

# Used to block processing when Arduino sends a "/off" command.
enable_processing = True

# Records whether the program has just started
state_uninitialized = True

# Welcome message
send_message(" * * * ", 3)
send_message("       ")

import random
import copy
while True:
    if enable_arduino:
        # Try to receive ASCIIMassage from Arduino
        try:
            line = ard.readline().rstrip().split()
            if line:
                command = line[0]

                # Off command sent by the Arduino to pause the processing.
                if command == b'/off':
                    if enable_processing:
                        if not state_uninitialized:
                            switch_off()
                        enable_processing = False

                # Reset states.
                elif command == b'/reset':
                    new_state.reset()
                    enable_processing = True

                # Connected wires.
                elif command == b'/conn':
                    try:
                        from_pin = int(line[1])
                        to_pin   = int(line[2])
                        new_state.connect(from_pin, to_pin)
                    except:
                        pass

                # State completed: update it.
                elif command == b'/done':
                    process_state()
                    state_uninitialized = False
        #            state.debug()
                    state.copy_from(new_state)
        except Exception as e:
            print("Exception: {}".format(e))
            pass

    if enable_processing:
        # Look if we need to emit a character.
        ct = time.time()
        if ct - start_time >= period:
            print("t={} ({})".format(ct, ct-start_time))
            generate_next()
            start_time = ct

server.shutdown()
print("\nDone.")
