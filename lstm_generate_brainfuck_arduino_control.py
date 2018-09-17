#!/usr/bin/python3
import serial
import syslog
import time
import argparse

from pythonosc import osc_message_builder
from pythonosc import udp_client

parser = argparse.ArgumentParser()
parser.add_argument("--serial-port", default="/dev/ttyUSB0", help="The serial port")
parser.add_argument("--baudrate", default=115200, help="The baudrate")

parser.add_argument("--ip", default="127.0.0.1", help="The ip of the OSC server")
parser.add_argument("--port", type=int, default=5005, help="The port the OSC server is listening on")

args = parser.parse_args()

client = udp_client.SimpleUDPClient(args.ip, args.port)
ard = serial.Serial(args.serial_port,args.baudrate,timeout=5)

# This is just a test.
temperature = 0.1
new_temperature = temperature

# Initialize.
#client.send_message("/neoism/start")

client.send_message("/neoism/sampling_mode", "softmax")
client.send_message("/neoism/n_best", 0)
client.send_message("/neoism/temperature", new_temperature)

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

def process_group(port_names, group, block_size, n_units, noise_level, lstm=False):
    global state
    from_i = 0
    lstm_offset = 0
    for port in port_names:
        connections = state.get_connections(port)
        port_info = state.get_info(port)
        arguments = [ group, from_i, block_size, 0, n_units]
        message_prefix = "/neoism/brain_"
        if lstm:
            message_prefix += "lstm_"
            arguments += [lstm_offset]

        if not connections:
            client.send_message(message_prefix + "cut", arguments)
        else:
            info = state.get_info(connections[0])
            if info["code"] == port_info["code"]: # correct port
                client.send_message(message_prefix + "restore", arguments)
            else:
                client.send_message(message_prefix + "noise", arguments + [noise_level])
#                speed_adjust += port_info["level"] - info["level"]
        # Special case: for LSTM layers only update the from_i variable at the end of offset cycles
        if lstm:
            lstm_offset += 1
            if lstm_offset >= 4:
                lstm_offset = 0
                from_i += block_size

def process_state():
#    print("processsing state")
    n_embeddings = 5
    n_characters = 56
    n_hidden_layer_1 = 64
    n_hidden_layer_2 = 1024

    speed_adjust = 0
    temperature_adjust = 0

    # Output layer.
#    print("Process output")
    process_group([ "31!", "32!", "33!" ], 7, int(n_hidden_layer_2 / 3), n_characters, 0.2)

    # LSTM layer #2
#    print("Process layer 2")
    process_group([ "21!", "22!", "23!", "24!", "25!", "26!", "27!", "28!" ], 4, int(n_hidden_layer_1 / 2), n_hidden_layer_2, 0.2, True)
    process_group([ "34!", "35!", "36!", "37!" ], 5, n_hidden_layer_2, n_hidden_layer_2, 0.2, True)

    # LSTM layer #1
#    print("Process layer 1")
    process_group([ "11!", "12!", "13!", "14!", "15!", "16!", "17!", "18!" ], 1, int(n_embeddings / 2), n_hidden_layer_1, 0.2, True)
    process_group([ "29!", "2A!", "2B!", "2C!" ], 2, n_hidden_layer_1, n_hidden_layer_1, 0.2, True)

    # Embeddings layer
#    print("Process embeddings")
    process_group([ "01!", "02!", "03!", "04!", "05!" ], 0, int(n_characters / 5), n_embeddings, 0.2)

    # # Recurrent layer #2
    # noise_level = 0.2
    # from_i = 0
    # group = 4
    # offset = 0
    # block_size = int(n_hidden_layer_1 / 2)
    # for port in [ "21!", "22!", "23!", "24!", "25!", "26!", "27!", "28!" ]:
    #     connections = state.get_connections(port)
    #     port_info = state.get_info(port)
    #     if not connections:
    #         client.send_message("/neoism/brain_lstm_cut", [ group, from_i, block_size, 0, n_hidden_layer_2, offset])
    #     else:
    #         info = state.get_info(connections[0])
    #         if info["code"] == port_info["code"]: # correct port
    #             client.send_message("/neoism/brain_lstm_restore", [ group, from_i, block_size, 0, n_hidden_layer_2, offset])
    #         else:
    #             client.send_message("/neoism/brain_lstm_noise", [ group, from_i, block_size, 0, n_hidden_layer_2, offset, noise_level])
    #             speed_adjust += port_info["level"] - info["level"]
    #     from_i += block_size
    #     offset = (offset + 1) % 4

    # Todo: check all inputs connected into inputs => increase temperature


state = State()
new_state = State()

import random
import copy
while True:
    try:
        # Receive ASCIIMassage from Arduino
        line = ard.readline().rstrip().split()
    except:
        continue
    if line:
        command = line[0]

        if command == b'/reset':
#            print("Reset state")
            new_state.reset()

        elif command == b'/conn':
            from_pin = int(line[1])
            to_pin   = int(line[2])
#            print("Connection: {} {}".format(from_pin, to_pin))
            new_state.connect(from_pin, to_pin)

        elif command == b'/done':
            print("=====")
            process_state()
            state.debug()
            state.copy_from(new_state)
#            print("Done")

    #client.send_message("/neoism/temperature")

    #     if temperature != new_temperature:
    #         client.send_message("/neoism/temperature", new_temperature)
    #         temperature = new_temperature

    # Apply state.


#    time.sleep(1)

exit()
