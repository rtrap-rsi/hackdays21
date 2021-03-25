import spacy
import json
import it_core_news_lg

nlp = it_core_news_lg.load()


# Add some custom stop words to the Spacy stop words list (words not needed for a semantic classification or clustering, like "the", "a", ecc)
def prepareNLP():
    stop_words = [""]
    for w in stop_words:
        nlp.vocab[w].is_stop = True


# Lemmatize a tokenized doc and remove pronuns (take the "root" of a word)
def lemmatize(doc):
    doc = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
    doc = u' '.join(doc)
    return nlp.make_doc(doc)


# Remove stop word and punctuation
def remove_stop_words(doc):
    # return token.text because we use these words in gensim
    doc = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
    return doc


def clean_text(text):
    return text


def getArticles():
    with open("articles.json", "r", encoding="utf-8") as articlesJson:
        return json.load(articlesJson)


def transcribe(articles):
    for article in articles:
        article["transcription"] = article["text"]


def tag(articles):
    for article in articles:
        cleaned_text = clean_text(article["transcription"])
        doc = nlp(cleaned_text)
        print(doc)


if __name__ == '__main__':
    # Add our preprocessing pipeline to spacy
    nlp.add_pipe(lemmatize, name='lemmatizer', after='ner')
    nlp.add_pipe(remove_stop_words, name="stopwords", last=True)

    articles = getArticles()
    transcribe(articles)
    tag(articles)
