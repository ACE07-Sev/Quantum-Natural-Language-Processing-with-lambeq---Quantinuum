from lambeq import BobcatParser, TketModel, remove_cups, IQPAnsatz, AtomicType, NumpyModel, spiders_reader
import lambeq.tokeniser.spacy_tokeniser
from pytket.extensions.qiskit import AerBackend
from discopy import Word
from discopy.rigid import Functor
from textblob import TextBlob
from cleantext import clean

from DepAnsatz import (Sim13Ansatz as Sim13)

print("packages are imported")


# Function for replacing the unknown word(s) with <unk> token
def replace(box):
    if box.name not in known_words_list:
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


# Function for printing the label based on the prediction array
def result_label(prediction):
    depressive_score = prediction[0][0]
    non_depressive_score = prediction[0][1]
    label = 0

    if depressive_score > non_depressive_score:
        label = 1
        return label
    elif depressive_score <= non_depressive_score:
        return label


"""
With this approach we can have both the similarity check of two sentences as well as their individual label, hence we 
won't have any loss of information since we trained the model to classify each label, not just which two labels are in
one category.
"""


def same_category(sentence1, sentence2):
    if result_label(sentence1) == result_label(sentence2) and result_label(sentence1) == 1:
        print("Both sentences are depressive")
        group = "Same Category Depressive"
        return group
    elif result_label(sentence1) == result_label(sentence2) and result_label(sentence1) == 0:
        print("Both sentences are non-depressive")
        group = "Same Category Non-depressive"
        return group
    elif result_label(sentence1) == 1 and result_label(sentence2) == 0:
        print("Sentence 1 is depressive and Sentence 2 is non-depressive")
        group = "Different Category"
        return group
    elif result_label(sentence1) == 0 and result_label(sentence2) == 1:
        print("Sentence 1 is non-depressive and Sentence 2 is depressive")
        group = "Different Category"
        return group


# initializing the parser
parser = spiders_reader
# parser = cups_reader

# initializing the Aer backend (uncomment only when using TketModel)
backend = AerBackend()

# backend configs
backend_config = {
    'backend': backend,
    'compilation': backend.default_compilation_pass(2),
    'shots': 8192
}

# Loading checkpoint model (300 epoch model)
# model = TketModel.from_checkpoint('C:/Users/elmm/Desktop/CQM/runs/Jun19_00-59-09_LT-ELMM-T-MY/model.lt', backend_config=backend_config)
model = NumpyModel.from_checkpoint('C:/Users/elmm/Desktop/CQM/runs/Aug23_02-33-20_LT-ELMM-T-MY/model.lt')

print(model.symbols)

# training dataset
train_labels, train_data = read_data('bc_train_data.txt')
# validation dataset
dev_labels, dev_data = read_data('bc_dev_data.txt')
# test dataset
test_labels, test_data = read_data('bc_test_data.txt')
# test sentence
test_sentence = ['I am really sad max .', 'I am very happy']

# initializing spacy tokenizer
tokeniser = lambeq.SpacyTokeniser()

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
# suffix tokenizing for test sentence
test_sentence = suffix_tokenizer(test_sentence)
test_sentence = fix_suffix_sentences(test_sentence, spacy_token=True)

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

for i in range(len(test_sentence)):
    # rewriting testing sentences for parser on sentence level (Use when not using Rewriter Object)
    # (Can use remove_dots_commas in either case)
    test_sentence[i] = remove_dots_commas(test_sentence[i])
    test_sentence[i] = remove_connector(test_sentence[i])
    test_sentence[i] = remove_auxiliary(test_sentence[i])
    test_sentence[i] = remove_determiner(test_sentence[i])

# parsing the dataset into sentence diagrams
train_diagrams = parser.sentences2diagrams(train_data)
dev_diagrams = parser.sentences2diagrams(dev_data)
test_diagrams = parser.sentences2diagrams(test_data)

# merging all diagrams into one
all_diagrams = train_diagrams + dev_diagrams + test_diagrams

# known symbols (330 words)
known_words = []
known_words_list = []
for i in range(len(all_diagrams)):
    known_words = [box.name for box in all_diagrams[i].boxes if isinstance(box, Word)]
    for j in range(len(known_words)):
        known_words_list.append(known_words[j])

print(known_words_list)
# new sample
new_diagram = parser.sentence2diagram(test_sentence[0])
new_diagram2 = parser.sentence2diagram(test_sentence[1])

# initializing the replace functor
replace_functor = Functor(ob=lambda x: x, ar=replace)

# tokenized diagram
replaced_diag = replace_functor(new_diagram)
replaced_diag.draw()
replaced_diag2 = replace_functor(new_diagram2)

# removing cups
diagram = remove_cups(replaced_diag)
diagram.draw()
diagram2 = remove_cups(replaced_diag2)

# initializing the ansatz for generating the circuit (1 layer, 3 qubits) (output : 1 qubit)
ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1, AtomicType.PREPOSITIONAL_PHRASE: 1}, n_layers=1,
                   n_single_qubit_params=3)

# initializing the ansatz for generating the circuit (2 layer, 3 qubits) (output : 1 qubit)
ansatz_original = Sim13({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1, AtomicType.PREPOSITIONAL_PHRASE: 1}, n_layers=5,
                        n_single_qubit_params=3)

# converting the diagram to a circuit
diagram = ansatz(diagram)
diagram2 = ansatz(diagram2)
# Circuit show
diagram.draw()

# predicting on new input
prediction = model.forward([diagram])
prediction2 = model.forward([diagram2])
result = prediction.tolist()
result2 = prediction2.tolist()
print(result)
print(result2)
same_category(result, result2)
