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
import unidecode
import sys

# Load file.
raw_text = codecs.open(args.text_file, "r", encoding='utf-8').read()

import unicodedata


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

raw_text = remove_accents(raw_text)
raw_text = raw_text.replace("ø", "o") #special case

# replace accents
#raw_text = unidecode.unidecode(raw_text)

# load ascii text and covert to lowercase
raw_text = raw_text.lower()

# Replace * * * * * style separators.
#raw_text = raw_text.replace("*", "")

# Replace line breaks with spaces and 2+line breaks with newlines.
raw_text = re.sub(r'\r\n', '\n', raw_text)
raw_text = re.sub(r' *\n', '\n', raw_text)
raw_text = re.sub(r'\n\n+', '\r', raw_text)
#raw_text = raw_text.replace("\n", " ")
raw_text = raw_text.replace("\r", "\n")

# Replace special/rare characters.
raw_text = raw_text.replace("--", "-")
raw_text = raw_text.replace(u"—",  "-")
raw_text = raw_text.replace(u" ",  " ") # replace non-breaking space by spaces
raw_text = raw_text.replace("(", "-")
raw_text = raw_text.replace(")", "-")

# Remove numbers
#raw_text = re.sub(r'\d', '', raw_text)

# Fix multiple white spaces problems.
raw_text = re.sub(r'\n +', '\n', raw_text)
raw_text = re.sub(r' +', ' ', raw_text)

# remove non-printable characters
#import string
#printable = set(string.printable)
#raw_text = ''.join(filter(lambda x: x in printable, raw_text))

# strange character
raw_text = raw_text.replace(u"\x0c",  "\n")

# remove newlines
#raw_text = raw_text.replace("\n", " ")
#raw_text = raw_text.replace("\r", " ")
#raw_text = raw_text.replace("\r\n", " ")

# remove very rare chars
raw_text = raw_text.replace("%", "")
raw_text = raw_text.replace("$", "")
raw_text = raw_text.replace("¢", "")
raw_text = raw_text.replace("©", "")
raw_text = raw_text.replace("+", "")
raw_text = raw_text.replace("<", "")
raw_text = raw_text.replace(">", "")
raw_text = raw_text.replace("=", "")
raw_text = raw_text.replace("|", "")
raw_text = raw_text.replace("[", "(")
raw_text = raw_text.replace("]", ")")
raw_text = raw_text.replace("{", "(")
raw_text = raw_text.replace("}", ")")
raw_text = raw_text.replace("_", "") # usually signifies italics
raw_text = raw_text.replace("*", "")
raw_text = raw_text.replace("/", "")
raw_text = raw_text.replace("–", "")
#raw_text = raw_text.replace(u"“",  "")
#raw_text = raw_text.replace(u"”",  "")
#raw_text = raw_text.replace("C", "")



# remove gibberish characters
raw_text = raw_text.replace(u"\x01",  "")
raw_text = raw_text.replace(u"\x04",  "")
raw_text = raw_text.replace(u"\x06",  "")
raw_text = raw_text.replace(u"\x08",  "")
raw_text = raw_text.replace(u"\x0f",  "")
raw_text = raw_text.replace(u"\x10",  "")
raw_text = raw_text.replace(u"\u2028",  "\n")


# Write to output.
output_file = codecs.open(args.output_file, "w+", encoding="utf-8")
output_file.write(raw_text)

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))

for c in chars:
  print(repr(c), " : ", str(raw_text.count(c)))
