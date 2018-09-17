# Load LSTM network and generate text
import argparse
import sys
from colr import color

from pythonosc import osc_message_builder, udp_client, dispatcher, osc_server

parser = argparse.ArgumentParser()

# OSC parameters
parser.add_argument("-rP", "--receive-port", type=int, default=5006, help="The port the OSC server is listening on")

args = parser.parse_args()

text_color = (255, 255, 255)

def receive_color(unused_addr, r, g, b):
    global text_color
    text_color = (r, g, b)

def receive_string(unused_addr, s):
#    print("Received : " + str(s))
    sys.stdout.write(color(s, fore=text_color, back=(0, 0, 0)))
    sys.stdout.flush()

dispatcher = dispatcher.Dispatcher()
dispatcher.map("/neoism/text", receive_string)
dispatcher.map("/neoism/color", receive_color)

server = osc_server.ThreadingOSCUDPServer( ("127.0.0.1", args.receive_port), dispatcher)
print("Serving on {}".format(server.server_address))
server.serve_forever()
