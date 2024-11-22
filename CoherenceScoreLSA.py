import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel

if __name__ == '__main__':
    data = pd.read_csv('clean_reviews.csv')
    clean_reviews = data['Clean_Reviews']
    tokens_clean_reviews = clean_reviews.apply(lambda x: x.split() if isinstance(x, str) else [])

    dictionary = Dictionary(tokens_clean_reviews)
    corpus = [dictionary.doc2bow(text) for text in tokens_clean_reviews]

    coherence_scores = []

    for i in range(1, 11):
        lsa_model = LsiModel(corpus=corpus, id2word=dictionary, num_topics=i, chunksize=200)

        coherence_model_lsi = CoherenceModel(model=lsa_model, texts=tokens_clean_reviews, dictionary=dictionary,
                                             coherence='c_v')
        coherence_lsi = coherence_model_lsi.get_coherence()

        coherence_scores.append(coherence_lsi)
        print(f"Coherence Score für {i} Themen: {coherence_lsi}")