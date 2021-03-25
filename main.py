import json
import glob
import numpy as np
import os
import re
from pprint import pprint
import it_core_news_lg
from gensim.models import CoherenceModel
from spacy import Language
from tqdm import tqdm
import gensim
import gensim.corpora as corpora

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
nlp = it_core_news_lg.load()
os.environ['MALLET_HOME'] = 'C:\\users\\rober\\src\\mallet-2.0.8'

# Add some custom stop words to the Spacy stop words list (words not needed for a semantic classification or clustering, like "the", "a", ecc)
def prepareNLP():
    stop_words = ["e","è","l'","i",'\n',"l’",'\n ','\n\n',"E","po'","“sono","d'","dell’"]
    for w in stop_words:
        nlp.vocab[w].is_stop = True

# Remove stop word and punctuation
@Language.component('stopwords')
def remove_stop_words(doc):
    # return token.text because we use these words in gensim
    doc = [re.sub('[\W]+', '', token.lemma_.lower()) for token in doc if token.is_stop != True and token.is_punct != True]
    return doc


def process_text(docs):
    doc_list = []
    # Iterates through each article in the corpus.
    for doc in tqdm(docs):
        # Passes that article through the pipeline and adds to a new list.
        doc_list.append(nlp(doc))
    return doc_list


def getArticles():
    articles=[]
    for file in glob.glob("cleaned/*",recursive=True):
        with open(file, "r", encoding="utf-8") as articlesJson:
            articles.append(json.load(articlesJson))
    return articles


def transcribe(articles):
    for article in articles:
        article["transcription"] = article["text"]

def test_lda_model(corpus, words,processed_articles):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=words,
                                                num_topics=30,
                                                random_state=2,
                                                passes=2,
                                                update_every=1,
                                                chunksize=500,
                                                alpha='auto',
                                                per_word_topics=True)

    pprint(lda_model.print_topics(num_words=10))

    print('Perplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_articles, dictionary=words, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Coherence Score: ', coherence_lda)
    lda_model.save("lda-30topics-10rs-2p-500cs-auto")

def test_mallet_model(corpus, words,processed_articles):
    malletPath = "C:\\Users\\rober\\src\\mallet-2.0.8\\bin\\mallet.bat"
    ldamallet = gensim.models.wrappers.LdaMallet(malletPath, corpus=corpus, num_topics=30, id2word=words)

    pprint(ldamallet.show_topics(formatted=False))

    coherence_model_mallet = CoherenceModel(model=ldamallet, texts=processed_articles, dictionary=words,
                                         coherence='c_v')
    coherence_mallet = coherence_model_mallet.get_coherence()
    print('Coherence Score: ', coherence_mallet)
    ldamallet.save("mallet-30topics")

def upload_processed():
    return np.load("processedArticles.npy", allow_pickle=True)

def save_processed(processed):
    np.save("processedArticles.npy",processed)

def get_clean_texts():
    return map(lambda x: x["transcription"].lower().replace("”", ""), articles)

if __name__ == '__main__':
    prepareNLP()
    # Add our preprocessing pipeline to spacy
    nlp.add_pipe("stopwords", last=True)

    loadProcessed = True
    if loadProcessed:
        processed_articles = upload_processed()
    else:
        articles = getArticles()
        transcribe(articles)
        texts = get_clean_texts()
        processed_articles = process_text(texts)
        save_processed(processed_articles)

    # Create the corpora for Gensim
    words = corpora.Dictionary(processed_articles)

    # Turns each document into a bag of words.
    corpus = [words.doc2bow(doc) for doc in processed_articles]

    # test_lda_model(corpus, words, processed_articles)
    test_mallet_model(corpus, words, processed_articles)
    ldamallet = gensim.models.wrappers.LdaMallet.load()