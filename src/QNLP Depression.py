import collections
import pickle
import sys
import warnings
import os
from random import shuffle

from textblob import TextBlob
from cleantext import clean

from discopy.tensor import Tensor
from discopy import Word
from discopy.rigid import Functor
import spacy

import matplotlib.pyplot as plt
import numpy as np
from numpy import random, unique
from lambeq import AtomicType, IQPAnsatz, remove_cups, NumpyModel, spiders_reader, cups_reader, stairs_reader, Rewriter
from lambeq import BobcatParser
from lambeq import Dataset
from lambeq import TreeReader, TreeReaderMode
from lambeq import QuantumTrainer, SPSAOptimizer
from lambeq import TketModel
from lambeq import SpacyTokeniser
from pytket.circuit.display import render_circuit_jupyter
from pytket.extensions.qiskit import AerBackend

from DepAnsatz import (Sim13Ansatz as Sim13)

print("packages are imported")

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
spacy.load('en_core_web_sm')

BATCH_SIZE = 30
EPOCHS = 300
EPSILON = sys.float_info.epsilon
SEED = 0

short_lists = [["TBH", "to be honest"], ["FYI", "for your information"], ["BRB", "be right back"]]


# Function for replacing low occuring word(s) with <unk> token
def replace(box):
    if isinstance(box, Word) and dataset.count(box.name) < 2:
        return Word('unk', box.cod, box.dom)
    return box


# Function for removing punctuations at sentence level
def remove_dots_commas(sentence):
    sentence = sentence.replace(",", "")
    sentence = sentence.replace(" .", "")
    return sentence


# Function for removing punctuations at sentence level for a list of sentences
def remove_dots_commas_sentences(list):
    for i in range(len(list)):
        list[i] = remove_dots_commas(list[i])
    return list


# Function for removing determiners at sentence level
def remove_determiner(sentence):
    sentence = sentence.replace("the ", "")
    sentence = sentence.replace("The ", "")
    return sentence


# Function for removing determiners at sentence level for a list of sentences
def remove_determiner_sentences(list):
    for i in range(len(list)):
        list[i] = remove_determiner(list[i])
    return list


# Function for removing auxiliary at sentence level
def remove_auxiliary(sentence):
    sentence = sentence.replace("I am", "I")
    sentence = sentence.replace("I 'm", "I")
    sentence = sentence.replace("I was", "I")
    sentence = sentence.replace("I were", "I")
    sentence = sentence.replace("you are", "you")
    sentence = sentence.replace("you were", "you")
    sentence = sentence.replace("you 're", "you")
    sentence = sentence.replace("he is", "he")
    sentence = sentence.replace("he was", "he")
    sentence = sentence.replace("he 's", "he")
    sentence = sentence.replace("she is", "she")
    sentence = sentence.replace("she was", "she")
    sentence = sentence.replace("she 's", "he")
    sentence = sentence.replace("they are", "they")
    sentence = sentence.replace("they 're", "they")
    sentence = sentence.replace("they were", "they")
    return sentence


# Fix suffix token at sentence level (you can use this function together with spacy tokenization)
def fix_suffix(sentence, spacy_token=False):
    if spacy_token is False:
        sentence = sentence.replace("'m", " am")
        sentence = sentence.replace("'s", " is")
        sentence = sentence.replace("'re", " are")
    else:
        sentence = sentence.replace("'m", "am")
        sentence = sentence.replace("'s", "is")
        sentence = sentence.replace("'re", "are")
    return sentence


# Fix suffix token at sentence level for a list of sentences
def fix_suffix_sentences(list, spacy_token):
    for i in range(len(list)):
        list[i] = fix_suffix(list[i], spacy_token)
    return list


# Function for removing auxiliary at sentence level for a list of sentences
def remove_auxiliary_sentences(list):
    for i in range(len(list)):
        list[i] = remove_auxiliary(list[i])
    return list


# Function for removing that connector at sentence level
def remove_connector(sentence):
    sentence = sentence.replace(" that", "")
    return sentence


# Function for removing that connector at sentence level for a list of sentences
def remove_connector_sentences(list):
    for i in range(len(list)):
        list[i] = remove_connector(list[i])
    return list


# Function for correcting dictation
def correct_dictation(sentence):
    sentence = TextBlob(sentence)
    result = sentence.correct()
    return result


# Function for correcting dication for a list of sentences
def correct_dication_sentences(list):
    for i in range(len(list)):
        list[i] = correct_dictation(list[i])
    return list


# Function for removing emojis
def remove_emoji(sentence):
    # removing emojis
    result = clean(sentence, no_emoji=True)
    # returning cleaned sentence
    return result


# Function for removing emojis for a list of sentences
def remove_emoji_sentences(list):
    for i in range(len(list)):
        list[i] = remove_emoji(list[i])
    return list


# reading dataset from a .txt file
def read_data(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = int(line[0])
            labels.append([t, 1 - t])
            sentences.append(line[1:].strip())
    return labels, sentences


# Function for SpaCy Tokenization
def suffix_tokenizer(list):
    # tokenizing each sentence in the list
    list = tokeniser.tokenise_sentences(list)
    # merging the tokenized words back into a sentence
    for i in range(len(list)):
        list[i] = ' '.join(list[i])
    # returning tokenized list
    return list


# train/test split
# training dataset
train_labels, train_data = read_data('bc_train_data.txt')
# validation dataset
dev_labels, dev_data = read_data('bc_dev_data.txt')
# test dataset
test_labels, test_data = read_data('bc_test_data.txt')

labels = train_labels + dev_labels + test_labels
data = train_data + dev_data + test_data

pairs = list(zip(labels, data))
random.seed(0)
random.shuffle(pairs)
print(len(pairs))

N_EXAMPLES = len(pairs)

# Shuffling datasets
train_labels, train_data = zip(*pairs[:round(N_EXAMPLES * 0.4)])
dev_labels, dev_data = zip(*pairs[round(N_EXAMPLES * 0.4):round(N_EXAMPLES * 0.8)])
test_labels, test_data = zip(*pairs[round(N_EXAMPLES * 0.8):])

# training set words (with repetition)
train_data_string = ' '.join(train_data)
train_data_list = train_data_string.split(' ')
# validation set words (with repetition)
dev_data_string = ' '.join(dev_data)
dev_data_list = dev_data_string.split(' ')
# test set words (with repetition)
test_data_string = ' '.join(test_data)
test_data_list = test_data_string.split(' ')

# initializing spacy tokenizer
tokeniser = SpacyTokeniser()

print(train_data)

"""
tokenize for words with suffix (We can use either the SpaCy tokenizer or the fix_suffix method)
Note : using both suffix_tokenizer and fix_suffix_sentences essentially allows the model to perform better
"""
# suffix tokenizing for training dataset
train_data = suffix_tokenizer(train_data)
train_data = fix_suffix_sentences(train_data, spacy_token=True)
# suffix tokenizing for validation dataset
dev_data = suffix_tokenizer(dev_data)
dev_data = fix_suffix_sentences(dev_data, spacy_token=True)
# suffix tokenizing for testing dataset
test_data = suffix_tokenizer(test_data)
test_data = fix_suffix_sentences(test_data, spacy_token=True)

for i in range(len(train_data)):
    # rewriting training sentences for parser on sentence level (Use when not using Rewriter Object)
    # (Can use remove_dots_commas in either case)
    train_data[i] = remove_dots_commas(train_data[i])
    train_data[i] = remove_connector(train_data[i])
    train_data[i] = remove_auxiliary(train_data[i])
    train_data[i] = remove_determiner(train_data[i])

for i in range(len(dev_data)):
    # rewriting validation sentences for parser on sentence level (Use when not using Rewriter Object)
    # (Can use remove_dots_commas in either case)
    dev_data[i] = remove_dots_commas(dev_data[i])
    dev_data[i] = remove_connector(dev_data[i])
    dev_data[i] = remove_auxiliary(dev_data[i])
    dev_data[i] = remove_determiner(dev_data[i])

for i in range(len(test_data)):
    # rewriting testing sentences for parser on sentence level (Use when not using Rewriter Object)
    # (Can use remove_dots_commas in either case)
    test_data[i] = remove_dots_commas(test_data[i])
    test_data[i] = remove_connector(test_data[i])
    test_data[i] = remove_auxiliary(test_data[i])
    test_data[i] = remove_determiner(test_data[i])

print(train_data)

# dataset words (with repetition)
dataset = train_data_list + dev_data_list + test_data_list
# list of all unique words in the dataset
unique_words = unique(dataset)
# frequency for each unique word
counter = collections.Counter(dataset)
print(counter)

# initializing the replace functor for UNK tokenization
replace_functor = Functor(ob=lambda x: x, ar=replace)

# initializing rewriter
rewriter = Rewriter(
    ['prepositional_phrase', 'determiner', 'auxiliary', 'curry', 'coordination', 'connector', 'preadverb', 'postadverb',
     'prepositional_phrase'])

# initializing the parser
parser = spiders_reader
# parser = BobcatParser(model_name_or_path='C:/Users/elmm/Desktop/CQM/Model')
# parser = cups_reader
# parser = stairs_reader


# parsing the dataset into sentence diagrams (requires Bobcat model to run locally)
raw_train_diagrams = parser.sentences2diagrams(train_data)
raw_dev_diagrams = parser.sentences2diagrams(dev_data)
raw_test_diagrams = parser.sentences2diagrams(test_data)

"""
Note : IF you are not using preprocessing at sentence level for rewrite rules besides remove_dots_commas, uncomment the
rewrite lines below with the normal_form() for normalizing the diagram. Given we are using spiders_reader, we will not
use the Rewriter object.
"""
# Rewriting the sentence diagrams
for i in range(len(raw_train_diagrams)):
    # raw_train_diagrams[i] = rewriter(raw_train_diagrams[i])
    # raw_train_diagrams[i] = raw_train_diagrams[i].normal_form()

    # Tokenizing low occuring words in training dataset
    raw_train_diagrams[i] = replace_functor(raw_train_diagrams[i])

for i in range(len(raw_dev_diagrams)):
    # raw_dev_diagrams[i] = rewriter(raw_dev_diagrams[i])
    # raw_dev_diagrams[i] = raw_dev_diagrams[i].normal_form()

    # Tokenizing low occuring words in validation dataset
    raw_dev_diagrams[i] = replace_functor(raw_dev_diagrams[i])

for i in range(len(raw_test_diagrams)):
    # raw_test_diagrams[i] = rewriter(raw_test_diagrams[i])
    # raw_test_diagrams[i] = raw_test_diagrams[i].normal_form()

    # Tokenizing low occuring words in test dataset
    raw_test_diagrams[i] = replace_functor(raw_test_diagrams[i])

# sample sentence diagram (entry 1)
raw_train_diagrams[0].draw()

# merging all diagrams into one for checking the new words
raw_all_diagrams = raw_train_diagrams + raw_dev_diagrams + raw_test_diagrams

# removing cups (after performing top-to-bottom scan of the word diagrams)
train_diagrams = [remove_cups(diagram) for diagram in raw_train_diagrams]
dev_diagrams = [remove_cups(diagram) for diagram in raw_dev_diagrams]
test_diagrams = [remove_cups(diagram) for diagram in raw_test_diagrams]

# sample sentence diagram (entry 1)
train_diagrams[0].draw()

# initializing the ansatz for generating the circuit (1 layer, 3 qubits) (output : 1 qubit)
ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1, AtomicType.PREPOSITIONAL_PHRASE: 1}, n_layers=1,
                   n_single_qubit_params=3)

# initializing the ansatz for generating the circuit (2 layer, 3 qubits) (output : 1 qubit)
ansatz_original = Sim13({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1, AtomicType.PREPOSITIONAL_PHRASE: 1}, n_layers=5,
                        n_single_qubit_params=3)

# train/test circuits
train_circuits = [ansatz(diagram) for diagram in train_diagrams]
dev_circuits = [ansatz(diagram) for diagram in dev_diagrams]
test_circuits = [ansatz(diagram) for diagram in test_diagrams]

# sample circuit diagram
train_circuits[0].draw(figsize=(9, 12))
train_circuit_tk = train_circuits[0].to_tk()
render_circuit_jupyter(train_circuit_tk)

# all circuits
all_circuits = train_circuits + dev_circuits + test_circuits

# initializing the Aer backend only when using TketModel
backend = AerBackend()
# backend configs
backend_config = {
    'backend': backend,
    'compilation': backend.default_compilation_pass(2),
    'shots': 8192
}

# initializing the Numpy Quantum Model model (TKET simulates a NISQ system and can be time consuming, hence we will use Numpy)
# model = TketModel.from_diagrams(all_circuits, backend_config=backend_config)
model = NumpyModel.from_diagrams(all_circuits)

# defining loss and accuracy (binary cross entropy)
loss = lambda y_hat, y: -np.sum(y * np.log(y_hat + EPSILON)) / len(y)
acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2

# initializing the trainer for the model
trainer = QuantumTrainer(
    model,
    loss_function=loss,
    epochs=EPOCHS,
    optimizer=SPSAOptimizer,  # Simultaneous Perturbation Stochastic Approximation optimizer
    optim_hyperparams={'a': 0.2, 'c': 0.06, 'A': 0.01 * EPOCHS},
    evaluate_functions={'acc': acc},
    evaluate_on_train=True,
    verbose='text',
    seed=SEED
)
# training dataset (Quantum batch optimization)
train_dataset = Dataset(train_circuits, train_labels, batch_size=BATCH_SIZE)
# validation dataset
val_dataset = Dataset(dev_circuits, dev_labels, shuffle=False)
# fitting the model (training on train and validation dataset)
trainer.fit(train_dataset, val_dataset, logging_step=12)

# plotting results
fig, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(10, 6))
ax_tl.set_title('Training set')
ax_tr.set_title('Development set')
ax_bl.set_xlabel('Iterations')
ax_br.set_xlabel('Iterations')
ax_bl.set_ylabel('Accuracy')
ax_tl.set_ylabel('Loss')

colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
ax_tl.plot(trainer.train_epoch_costs, color=next(colours))
ax_bl.plot(trainer.train_results['acc'], color=next(colours))
ax_tr.plot(trainer.val_costs, color=next(colours))
ax_br.plot(trainer.val_results['acc'], color=next(colours))
plt.show()

# Model accuracy on test dataset
test_acc = acc(model(test_circuits), test_labels)
train_acc = acc(model(train_circuits), train_labels)
dev_acc = acc(model(dev_circuits), dev_labels)
cumulative_acc = (train_acc + dev_acc + test_acc) / 3
print('Test accuracy:', test_acc)
print("Model summary")
print("Train accuracy : ", train_acc)
print("Validation accuracy : ", dev_acc)
print("Test accuracy: ", test_acc)
print("Cummulative accuracy : ", cumulative_acc)
