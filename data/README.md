# Methodology

The datasets are made of sentences and their respective binary labels based on being depressive or non-depressive.

1  life is very depressing .
1  I cry in my sleep .
1  he was in a confused state of mind .
1  depression can be traced to holding in anger .
1  it is a sad and depressed world .
1  woman began to cry .
1  she was filled with anger and sorrow .
0  I hope you two are happy together .
0  you are happy .
0  I was very happy .
0  I am very happy .
0  she was happy .
0  she would like to have so much happiness .
.
.
.

The methodology for preparing this dataset was to provide a sentences which allowed the model to perform the sentiment analysis through means of syntactic sequence and
high occuring words in each label. We have provided over 300 sentences with a variety of themes and scenarios, all of which share the feature of including a single or
multiple words which can summarize the sentiment of the sentence, "depressive", "happy", "sad", "like to", etc. 

Given the complexities of Language, we have chosen Lambeq with two set(s) of readers/parsers to perform the NLP process through a dependency feature :

1) Sequence Independent : Readers like Spiders_reader which is essentially a Bag Of Words reader, allowing for a sequence independent parsing.
2) Sequence Dependent : Readers like Bobcat Parser, Cups_reader, stairs_reader and Tree_reader, each of which has had its own flaws and favors.

We have finalized on one reader per category. Given the nature of the categories, the sentences have been designed to be as standard to classic MNIST datasets as possible
given the impact of tokenization (UNK tokenization and its impact on crucial but low occuring words which may cause low accuracy in model), and rewriting.
