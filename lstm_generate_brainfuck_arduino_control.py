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
#client.send_message("/readings/start")

client.send_message("/readings/sampling_mode", "softmax")
client.send_message("/readings/n_best", 0)
client.send_message("/readings/temperature", new_temperature)

import random
while True:
    # Receive ASCIIMassage from Arduino
    line = ard.readline().rstrip().split()
    command = line[0]

    if command == b'/reset':
        print("Reset state")
        new_temperature = 0.1

    elif command == b'/conn':
        from_pin = int(line[1])
        to_pin   = int(line[2])
        print("Connection: {} {}".format(from_pin, to_pin))
        new_temperature += 0.2

    elif command == b'/done':
        if temperature != new_temperature:
            client.send_message("/readings/temperature", new_temperature)
            temperature = new_temperature

#    time.sleep(1)

exit()
