---
title: "nlp"
date: 2018-12-23T15:44:02+01:00
draft: false
categories: ["Machine learning"]
tags: []
---


## 1. What is nlp and why should you care?

* NLP or **natural language processing** (*not* neuro-linguistic programming. This disambiguation makes searching on the internet a real pain) is a group of methods/algorithms that deal with **text data**.
Sounds like text mining? The former concentrates more on predictive modeling, while the latter on exploratory data analysis. In practice some of the algorithms may be used both for prediction and exploration (LDA, for example). 

* As a data scientist you should care, because text data contains a lot of useful and valuable information, so declining it just because you don't know how deal with it would be a huge waste ;)

## 2. NLP subjects and how to approach them

### Basic concepts

Basic concepts are: document, corpus, vector and model. You'll find their definitions [here](https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html#core-concepts).

### Preprocessing

The input data is usually very messy. Just as in traditional machine learning (I refer here to classical, matrix-transformable tabular format) the data must be 'clean' so the results would not come out messy as well. What does 'clean data' in nlp stand for? In general having run through the following steps should give us a fairly usable, valid dataset:

- **tokenization** - an absolutely basic concept, yet surprisingly simple. In practice tokenization stands for dividing your text into a list of words (sometimes pairs or triples of words, entire sentences or even syllables). A var brief tutorial should be enough, e.g. this one: <iframe width="800" height="449" src="https://www.youtube.com/embed/nxhCyeRR75Q?list=PLIG2x2RJ_4LTF-IIu7-J3y_yg8LRe1WZq" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

- excluding **stop words** - the most popular approach (yet not the best one) is to find a list of stop words for a particular language on the Internet. But... a better way is to exclude specific parts of speech, e.g. conjunctions. To do this, you have to tag each word with its part of speech (for Polish [morfeusz](http://morfeusz.sgjp.pl/) seems to be the best choice, with [spaCy](https://spacy.io/usage/spacy-101#annotations-pos-deps)'s API, so again I recommend [Natural Language Processing and Computational Linguistics: A practical guide to text analysis with Python, Gensim, spaCy, and Keras](https://www.amazon.com/Natural-Language-Processing-Computational-Linguistics-ebook/dp/B07BWH779J)).

- **stemming** and **lemmatization** - which of these you should use depends on the language that you are working on. For example for Polish (which is my first language) lemmatization works far better than stemming and the best tool I've come across so far is [morfeusz](http://morfeusz.sgjp.pl/) with its excellent [documentation](http://download.sgjp.pl/morfeusz/Morfeusz2.pdf), which introduces various quirks of the Polish language thoroughly. However the Morfeusz's API is not so user friendly (IMHO), so I recommend using [spaCy](https://spacy.io/usage/spacy-101#annotations-pos-deps)'s API to Morfeusz instead. A perfect place to begin you adventure with spaCy is [Natural Language Processing and Computational Linguistics: A practical guide to text analysis with Python, Gensim, spaCy, and Keras](https://www.amazon.com/Natural-Language-Processing-Computational-Linguistics-ebook/dp/B07BWH779J).

- last but not least, **regex**. To be honest, even though most nlp libraries provide regex functionalities, I stick to the old good Python's [re](https://docs.python.org/3/library/re.html) package because of its great performance thanks to [re.compile()](https://docs.python.org/3/library/re.html#re.compile) philosophy and huge popularity and support. And easiness of use. And support for standard [regex](https://cheatography.com/davechild/cheat-sheets/regular-expressions/) syntax.

### Transforming

- **BOW** - the simplest idea of transforming words into numbers by counting the occurrences of each word in a document.

- **tf-idf** (best resource: An Introduction to Information Retrieval, Manning, Raghavan, Shutze, chapter 6: Scoring, term weighting and the vector space model) - for a good start you should familiarize yourself with how unstructured data, especially text data is represented in mathematical terms. For this, tf-idf is used, which being a rather straightforward concept, enables representing documents as vectors. In result it becomes possible to calculate resemblance (proximity) of two documents by simply applying cosine measure (a close relative to Pearson's correlation coefficient).

- **word2vec** - representing each word as a dense vector, which contains meaning of the word. The most famous example is the equation king - man + woman = queen, presented in [word2vec paper](https://arxiv.org/abs/1301.3781), which appears in 100% articles about word2vec, so I had to mention it as well. However the original paper is a little bit obscure and does not contentrate on mathematics too much, so I recommend reading [word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) instead and definitely watching [a lecture from Stanford conducted by legendary Christopher D. Manning](https://www.youtube.com/watch?v=HnNJc1AcF14&ab_channel=ClintJennings).  Even though word2vec is based on a shallow neural network, I don't use any of famous NN libraries (tensorflow, pytorch, keras) for obtaining predictions, but simply call gensim's [word2vec](https://radimrehurek.com/gensim/models/word2vec.html) model.

### Exploration

- **doc2vec** is pretty much a word2vec model with a minor tweak (adding a particular sentence's id as if it were another word). Official [paper](https://arxiv.org/abs/1405.4053) is rather approachable, but running through examples from [gensim's doc2vec model](https://radimrehurek.com/gensim/models/doc2vec.html) will provide you with a good intuition on what the capabilities of this algorithm are: you can very easily measure similarities between various documents.

- **LDA**

    Probably the most popular algorithm for topic modeling is LDA (Latent Dirichlet Allocation), however before deep diving into papers on this particular subject, you should consider first whether you already have appropriate background. The easy way to get familiar with LDA is starting with:
    - latent semantic indexing (An Introduction to Information Retrieval, Manning, Raghaven, Schutze, chapter 18: Matrix decomposition and latent semantic indexing). Understanding LSA (Latent Semantic Indexing) is an excellent step towards understanding LDA, however at first the concept may seem a little obscure, especially if you haven't been using linear algebra for a while. These resources should be helpful:

    - singular value decomposition (SVD) and low-rank matrix approximations (CS168: The Modern Algorithmic Toolbox, Lecture #9 [link](https://web.stanford.edu/class/cs168/l/l9.pdf))

    - besides SVD, it's good to remember how PCA actually works. I've always been finding it hard to understand *why* eigenvalues and eigenvectors are the best choices in PCA, and by *why* I mean that this should be as obvious to me as arithmetic mean. Excellent resources are here: [one](https://web.stanford.edu/class/cs168/l/l7.pdf) and [two](https://web.stanford.edu/class/cs168/l/l8.pdf), also from The Modern Algorithmic Toolbox. Chapeau bas, Stanford University!

    - having read about SVD, the Python package that I find best for topic modelling is [gensim](https://radimrehurek.com/gensim/) with its wonderful [tutorials](https://radimrehurek.com/gensim/auto_examples/index.html#documentation).

    Armed with tha basic intuition on LDA, you may deep dive into more advanced resources:

    - [The Little Book of LDA](https://ldabook.com/background.html)

    - [long tutorial from medium](https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24)

    - [LDA's original paper](https://www.seas.harvard.edu/courses/cs281/papers/blei-ng-jordan-2003.pdf) - for crazy-braves only.

    INHO, depp understanding of LDA is not necessary to make use of its functionalities and advantages.

### Learning (classification)

- **LSTM** - for me the best resource to understand those was [this article from Stanford University](https://web.stanford.edu/class/cs379c/archive/2018/class_messages_listing/content/Artificial_Neural_Network_Technology_Tutorials/OlahLSTM-NEURAL-NETWORK-TUTORIAL-15.pdf).

- **BERT** - I have a very basic intuition on how this works thanks to [BERT's original paper](https://arxiv.org/abs/1810.04805) and I just use it :) with [BERT-pytorch](https://github.com/codertimo/BERT-pytorch).

### Other interesting resources

- [Text mining in R](https://www.tidytextmining.com/) - basic concepts in text mining, introduction to tidytext package and LDA

- [example nlp usage in Python](https://towardsdatascience.com/gentle-start-to-natural-language-processing-using-python-6e46c07addf3) - a towardsdatascience article

