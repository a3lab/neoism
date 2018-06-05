# Load LSTM network and generate text
import argparse
import sys

from pythonosc import osc_message_builder, udp_client, dispatcher, osc_server

parser = argparse.ArgumentParser()

# OSC parameters
parser.add_argument("-rP", "--receive-port", type=int, default=5006, help="The port the OSC server is listening on")

args = parser.parse_args()

def receive_string(unused_addr, s):
#    print("Received : " + str(s))
    sys.stdout.write(s)
    sys.stdout.flush()

dispatcher = dispatcher.Dispatcher()
dispatcher.map("/readings/string", receive_string)

server = osc_server.ThreadingOSCUDPServer( ("127.0.0.1", args.receive_port), dispatcher)
print("Serving on {}".format(server.server_address))
server.serve_forever()
