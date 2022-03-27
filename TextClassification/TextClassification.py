import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb
# num_words=10000 -> take the words that are the 10000s most frequent
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)
print(train_data[0]) #integer encoded words

#find the mapping for the words
#gives a tuples - (key, values) that has the keys-mappings to figure out what those integers mean
word_index = data.get_word_index()
word_index = {k:(v+3) for k, v in word_index.items()} #start at +3, because 3 keys are special characters

#assign my own values for the padding, start, unknown, unused; so if we get values that are not valid
#we can assign them to the 4 defined
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

#swap all the values and keys, so the dictionaries have the keys=words first and values=integers second
reverse_word_index = dict([(value, key) for(key, value) in word_index.items()])

#make everything in train_data, and test_data same length of max 250 (by adding padding where needed and trimming where the length is > 250)
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

#print(len(test_data), len(test_data))

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text]) #return all keys=words

print(decode_review(test_data[0]))
#print(len(test_data[0]), len(test_data[1]))

#define the model; the output neuron that will tell us if the review is good or bad (so 0 or 1)
model = keras.Sequential()
model.add(kera.layers.Embedding(10000, 16))
model.add(kera.layers.GlobalAveragePooling1D())
model.add(kera.layers.Dense(16, activation="relu"))
model.add(kera.layers.Dense(1, activation="sigmoid"))

model.summary()
