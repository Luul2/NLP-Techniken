import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LdaMulticore

if __name__ == '__main__':
    data = pd.read_csv('clean_reviews.csv')
    clean_reviews = data['Clean_Reviews']
    tokens_clean_reviews = clean_reviews.apply(lambda x: x.split() if isinstance(x, str) else [])

    dictionary = Dictionary(tokens_clean_reviews)
    corpus = [dictionary.doc2bow(text) for text in tokens_clean_reviews]

    coherence_scores = []

    for i in range(1, 11):
        lda_model = LdaMulticore(corpus=corpus,id2word=dictionary,num_topics=i,random_state=100,passes=10,chunksize=200,
                                 workers=10)

        coherence_model_lda = CoherenceModel(model=lda_model, texts=tokens_clean_reviews, dictionary=dictionary,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()

        coherence_scores.append(coherence_lda)
        print(f"Coherence Score für {i} Themen: {coherence_lda}")
