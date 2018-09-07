#!/usr/bin/python
import serial
import syslog
import time

#The following line is for serial over GPIO
port = '/dev/ttyUSB0'
baudrate = 9600

ard = serial.Serial(port,baudrate,timeout=5)

while True:
    line = ard.readline().rstrip().split()
    command = line[0]
    if command == b'/reset':
        print("Reset state")
    elif command == b'/conn':
        from_pin = int(line[1])
        to_pin   = int(line[2])
        print("Connection: {} {}".format(from_pin, to_pin))

exit()
