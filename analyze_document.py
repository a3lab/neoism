# -*- coding: utf-8 -*-
# Source: http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("text_file", type=str, help="The file containing the original text")
parser.add_argument("-N", "--n-words", type=int, default=100, help="Output only top N words")
parser.add_argument("-b", "--begin-char", type=int, default=0, help="Beginning character")
parser.add_argument("-e", "--end-char", type=int, default=-1, help="Ending character")

args = parser.parse_args()

import numpy
import os
import nltk

def print_frequencies_csv(dist, n, separator=''):
    top = dist.most_common(n)
    for (item, freq) in top:
        if (isinstance(item, tuple)):
            print("{}\t{}\t".format(separator.join(item), freq))
        else:
            print("{}\t{}\t".format(item, freq))

# load ascii text and covert to lowercase
raw_text = open(args.text_file).read()
raw_text = raw_text.lower()

n_tokens = args.n_words

begin_char = args.begin_char
end_char = args.end_char
if (end_char < 0):
    end_char = len(raw_text)-1

raw_text = raw_text[begin_char:end_char]

# create word tokenization
words = nltk.tokenize.word_tokenize(raw_text)

# create frequency distribution
freq = nltk.FreqDist(raw_text)
freq_words = nltk.FreqDist(words)

print("Character frequencies ===")
print_frequencies_csv(freq, 1000)

print("Bigrams =================")
bigrams = nltk.bigrams(raw_text)
fdist = nltk.FreqDist(bigrams)
print_frequencies_csv(fdist, n_tokens)

print("Trigrams ================")
trigrams = nltk.trigrams(raw_text)
fdist3 = nltk.FreqDist(trigrams)
print_frequencies_csv(fdist3, n_tokens)

print("Word frequencies ========")
print_frequencies_csv(freq_words, n_tokens)

print("Word bigrams ============")
bigrams = nltk.bigrams(words)
fdist = nltk.FreqDist(bigrams)
print_frequencies_csv(fdist, n_tokens, ' ')

print("Word trigrams ===========")
trigrams = nltk.trigrams(words)
fdist3 = nltk.FreqDist(trigrams)
print_frequencies_csv(fdist3, n_tokens, ' ')
