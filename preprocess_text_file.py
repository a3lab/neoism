# coding=utf-8
# Note: first use http://www.utf8-chartable.de/unicode-utf8-table.pl?unicodeinhtml=hex
#       to replace strange/unique letters by hand
import argparse

parser = argparse.ArgumentParser(description="Preprocesses text files to make them easier to train")
parser.add_argument("text_file", type=str, help="The file containing the original text")
parser.add_argument("output_file", type=str, help="The output file")

args = parser.parse_args()

import re
import codecs

# Load file.
raw_text = codecs.open(args.text_file, "r", encoding='utf-8').read()

# load ascii text and covert to lowercase
raw_text = raw_text.lower()

# Replace * * * * * style separators.
raw_text = raw_text.replace("*", "")

# Replace line breaks with spaces and 2+line breaks with newlines.
raw_text = re.sub(r'\r\n', '\n', raw_text)
raw_text = re.sub(r' *\n', '\n', raw_text)
raw_text = re.sub(r'\n\n+', '\r', raw_text)
raw_text = raw_text.replace("\n", " ")
raw_text = raw_text.replace("\r", "\n")

# Replace special/rare characters.
raw_text = raw_text.replace("--", "-")
raw_text = raw_text.replace(u"—",  "-")
raw_text = raw_text.replace(u" ",  " ") # replace non-breaking space by spaces
raw_text = raw_text.replace("(", "-")
raw_text = raw_text.replace(")", "-")

raw_text = re.sub(r'\d', '', raw_text) # remove numbers

# Remove rare characters.
raw_text = raw_text.replace("_", "") # usually signifies italics
raw_text = raw_text.replace("*", "")
raw_text = raw_text.replace("/", "")
raw_text = raw_text.replace(u"“",  "")
raw_text = raw_text.replace(u"”",  "")

# Fix multiple white spaces problems.
raw_text = re.sub(r'\n +', '\n', raw_text)
raw_text = re.sub(r' +', ' ', raw_text)

# Write to output.
output_file = codecs.open(args.output_file, "w+", encoding="utf-8")
output_file.write(raw_text)

# load ascii text and covert to lowercase
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))

for c in chars:
  print c + " : " + str(raw_text.count(c))
