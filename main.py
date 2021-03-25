import spacy
import json
import glob
import it_core_news_lg
from spacy import Language
from tqdm import tqdm
import gensim
import gensim.corpora as corpora
from spacy.lang.it.stop_words import STOP_WORDS

nlp = it_core_news_lg.load()


# Add some custom stop words to the Spacy stop words list (words not needed for a semantic classification or clustering, like "the", "a", ecc)
def prepareNLP():
    stop_words = ["e","è","l'","i",'\n',"l’",'\n ','\n\n']
    for w in stop_words:
        nlp.vocab[w].is_stop = True



# # Lemmatize a tokenized doc and remove pronuns (take the "root" of a word)
# @Language.component('lemmatize')
# def lemmatize(doc):
#     doc = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
#     doc = u' '.join(doc)
#     return nlp.make_doc(doc)


# Remove stop word and punctuation
@Language.component('stopwords')
def remove_stop_words(doc):
    # return token.text because we use these words in gensim
    doc = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
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


if __name__ == '__main__':
    prepareNLP()
    # Add our preprocessing pipeline to spacy
    nlp.add_pipe("stopwords", last=True)

    articles = getArticles()
    transcribe(articles)
    processed_articles =process_text(map(lambda x: x["transcription"], articles))
    print(processed_articles[0])
    # Create the corpora for Gensim
    words = corpora.Dictionary(processed_articles)

    # Turns each document into a bag of words.
    corpus = [words.doc2bow(doc) for doc in processed_articles]
    lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                           id2word=words,
                                           num_topics=10,
                                           random_state=2,
                                           workers=7,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

    print(lda_model.print_topics(num_words=10))