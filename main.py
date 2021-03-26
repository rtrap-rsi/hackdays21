# ----------------------------------------- LIBRARIES ----------------------------------------- #
import json
import glob
import numpy as np
import os
import re
import yaml
from pprint import pprint
import spacy
from gensim.models import CoherenceModel
from spacy import Language
from tqdm import tqdm
import gensim
import gensim.corpora as corpora
from pathlib import Path
import logging as log
import matplotlib.pyplot as plt

# ----------------------------------------- CONFIG ----------------------------------------- #

with open('config.yaml') as file:
    configs = yaml.full_load(file)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
os.environ['MALLET_HOME'] = str(Path(configs['malletHome']))

if configs['logLevel'] != "None":
    log.basicConfig(level=configs['logLevel'])

nlp = spacy.load(configs['spacy']['model'])


# ----------------------------------------- FUNCTIONS ----------------------------------------- #

# Add some custom stop words to the Spacy stop words list (words not needed for a semantic classification or clustering, like "the", "a", ecc)
def prepareNLP():
    stop_words = ["e", "è", "l'", "i", '\n', "l’", '\n ', '\n\n', "E", "po'", "“sono", "d'", "dell’", "della", "un"]
    print("Updating Spacy stopwords list with customs: ")
    print(stop_words)
    for w in stop_words:
        nlp.vocab[w].is_stop = True


# Clean text before the spacy pipeline, just do everything lowercase and little workaround for an error with the mallet implementation (.replace("”", ""))
def get_clean_texts(articles):
    print("Cleaning texts")
    return map(lambda x: x["transcription"].lower().replace("”", ""), articles)


# Remove stop word and punctuation from a list of spacy Doc
@Language.component('stopwords')
def remove_stop_words(doc):
    # return token.lemma_ because we use these words in gensim
    doc = [re.sub('[\W]+', '', token.lemma_.lower()) for token in doc if
           token.is_stop != True and token.is_punct != True]
    return doc


# Return a list of Doc processed spacy object from a list of texts
def process_text(docs):
    print("Passing documents through Spacy pipeline")
    doc_list = []
    # Iterates through each article in the corpus.
    for doc in tqdm(docs):
        # Passes that article through the pipeline and adds to a new list.
        doc_list.append(nlp(doc))
    return doc_list


# Get articles from a folder, one article per file stored as json
def getArticles():
    pprint(f'Getting articles from {configs["dataset"]}')
    articles = []
    for file in glob.glob(f'{configs["dataset"]}/*', recursive=True):
        with open(file, "r", encoding="utf-8") as articlesJson:
            articles.append(json.load(articlesJson))
    return articles


# This is a fake step, in our workflow we should take audios, transcribe and then tagging. For hackdays purpose we just trained our model on RSI articles
def transcribe(articles):
    for article in articles:
        article["transcription"] = article["text"]


# Test of the Latend Dirichlet Allocation statistical model
def test_lda_model(corpus, words, processed_articles):
    print("\nRunning LDA model")
    if configs['topicModel']['loadFromFile'] == '':
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=words,
                                                    num_topics=configs['topicModel']['lda']['num_topics'],
                                                    random_state=configs['topicModel']['lda']['random_state'],
                                                    passes=configs['topicModel']['lda']['passes'],
                                                    update_every=configs['topicModel']['lda']['update_every'],
                                                    chunksize=configs['topicModel']['lda']['chunksize'],
                                                    alpha=configs['topicModel']['lda']['alpha'],
                                                    per_word_topics=True)
    else:
        lda_model = gensim.models.ldamodel.LdaModel.load(configs['topicModel']['loadFromFile'])

    pprint(lda_model.print_topics(num_words=20))

    print('Perplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
    if configs['topicModel']['calculateCoherence']:
        print('Calculating coherence...')
        coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_articles, dictionary=words,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('Coherence Score: ', coherence_lda)
    return lda_model


# Test of the Mallet LDA model
def test_mallet_model(corpus, words, processed_articles):
    print("\nRunning Mallet model")
    malletPath = "C:\\Users\\rober\\src\\mallet-2.0.8\\bin\\mallet.bat"
    if configs['topicModel']['loadFromFile'] == '':
        ldamallet = gensim.models.wrappers.LdaMallet(malletPath, corpus=corpus,
                                                     num_topics=configs['topicModel']['mallet']['num_topics'],
                                                     id2word=words, alpha=configs['topicModel']['mallet']['alpha'],
                                                     iterations=configs['topicModel']['mallet']['iterations'])
    else:
        ldamallet = gensim.models.wrappers.LdaMallet.load(configs['topicModel']['loadFromFile'])
    pprint(ldamallet.print_topics())
    if configs['topicModel']['calculateCoherence']:
        print('Calculating coherence...')
        coherence_model_mallet = CoherenceModel(model=ldamallet, texts=processed_articles, dictionary=words,
                                                coherence='c_v')
        coherence_mallet = coherence_model_mallet.get_coherence()
        print('Coherence Score: ', coherence_mallet)
    return ldamallet


# Test of the HDP model
def test_hdp_model(corpus, words, processed_articles):
    print("\nRunning HDP model")
    if configs['topicModel']['loadFromFile'] == '':
        hdp_model = gensim.models.hdpmodel.HdpModel(corpus=corpus,
                                                    id2word=words)
    else:
        hdp_model = gensim.models.hdpmodel.HdpModel.load(configs['topicModel']['loadFromFile'])

    pprint(hdp_model.print_topics(num_words=10))

    if configs['topicModel']['calculateCoherence']:
        print('Calculating coherence...')
        coherence_model_lda = CoherenceModel(model=hdp_model, texts=processed_articles, dictionary=words,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('Coherence Score: ', coherence_lda)
    return hdp_model


def load_processed():
    return np.load("processedArticles.npy", allow_pickle=True)


def save_processed(processed):
    print("Saving processed articles")
    np.save("processedArticles.npy", processed)


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    malletPath = "C:\\Users\\rober\\src\\mallet-2.0.8\\bin\\mallet.bat"
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(malletPath, corpus=corpus, num_topics=num_topics, id2word=dictionary,
                                                 alpha=configs['topicModel']['mallet']['alpha'])
        model_list.append(model)

        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# ----------------------------------------- MAIN ----------------------------------------- #

if __name__ == '__main__':
    print("--- Starting topic modeling ---")
    prepareNLP()
    # Add our preprocessing pipeline to spacy
    nlp.add_pipe("stopwords", last=True)

    loadProcessed = configs['spacy']['useSaved']
    if loadProcessed:
        print("Loading saved Spacy processed articles")
        processed_articles = load_processed()
    else:
        articles = getArticles()
        transcribe(articles)
        texts = get_clean_texts(articles)
        processed_articles = process_text(texts)
        save_processed(processed_articles)

    # Create the corpora for Gensim
    print("Create Gensim corpora from articles")
    words = corpora.Dictionary(processed_articles)

    # Turns each document into a bag of words.
    print("Turns documents into a bag of words")
    corpus = [words.doc2bow(doc) for doc in processed_articles]

    model = None
    modelName = configs['topicModel']['use']
    if modelName == 'lda':
        model = test_lda_model(corpus, words, processed_articles)
    elif modelName == 'mallet':
        model = test_mallet_model(corpus, words, processed_articles)
    elif modelName == 'hdp':
        model = test_hdp_model(corpus, words, processed_articles)
    else:
        print(f"No model named {modelName} found")

    if configs['testDocument']:
        test_articles = []
        with open(Path(configs['testDocument']), "r", encoding="utf-8") as articleJson:
            test_articles.append(json.load(articleJson))
        transcribe(test_articles)
        test_texts = get_clean_texts(test_articles)
        test_processed_articles = process_text(test_texts)
        test_words = corpora.Dictionary(test_processed_articles)
        test_corpus = [words.doc2bow(doc) for doc in test_processed_articles]
        print(model[test_corpus[0]])
        doc_topics = model[test_corpus[0]]
        most_relevant_doc_topic_id = max(doc_topics,key=lambda i:i[1])[0]
        # print(most_relevant_doc_topic_id)
        print(model.show_topic(most_relevant_doc_topic_id, topn=20))

        # print("Search best num topics")
        # model_list, coherence_values = compute_coherence_values(dictionary=words, corpus=corpus,
        #                                                         texts=processed_articles, start=2, limit=40, step=6)
        # print(coherence_values)

    if configs['topicModel']['saveAs'] and model != None and configs['topicModel']['loadFromFile'] == '':
        print('Saving topic model...')
        model.save(configs['topicModel']['saveAs'])
        print('Topic model saved!')
