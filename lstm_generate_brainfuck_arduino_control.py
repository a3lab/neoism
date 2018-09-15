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

def process_state():
    n_embeddings = 5
    n_characters = 56
    n_hidden_layer_1 = 64
    n_hidden_layer_2 = 1024

    speed_adjust = 0
    temperature_adjust = 0

    # Output layer.
    noise_level = 0.2
    from_i = 0
    block_size = int(n_hidden_layer_2 / 3)
    for port in [ "31!", "32!", "33!" ]:
        connections = state.get_connections(port)
        port_info = state.get_info(port)
        if not connections:
            client.send_message("/neoism/brain_cut", [ 7, from_i, block_size, 0, n_characters])
        else:
            info = state.get_info(connections[0])
            if info["code"] == port_info["code"]: # correct port
                client.send_message("/neoism/brain_restore", [ 7, from_i, block_size, 0, n_characters])
            else:
                client.send_message("/neoism/brain_noise", [ 7, from_i, block_size, 0, n_characters, noise_level])
                speed_adjust += port_info["level"] - info["level"]
        from_i += block_size

    # Todo: check all inputs connected into inputs => increase temperature


state = State()
new_state = State()

import random
import copy
while True:
    # Receive ASCIIMassage from Arduino
    line = ard.readline().rstrip().split()
    if line:
        command = line[0]

        if command == b'/reset':
            print("Reset state")
            new_state.reset()

        elif command == b'/conn':
            from_pin = int(line[1])
            to_pin   = int(line[2])
            print("Connection: {} {}".format(from_pin, to_pin))
            new_state.connect(from_pin, to_pin)

        elif command == b'/done':
            print("=====")
            process_state()
            state.debug()
            state.copy_from(new_state)
            print("Done")

    #client.send_message("/neoism/temperature")

    #     if temperature != new_temperature:
    #         client.send_message("/neoism/temperature", new_temperature)
    #         temperature = new_temperature

    # Apply state.


#    time.sleep(1)

exit()
