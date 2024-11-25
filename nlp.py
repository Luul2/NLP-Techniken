import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

data = pd.read_csv('reviews.csv')
reviews = data['Review_Text']

nltk.download('all')


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word.isalpha() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


data['Clean_Reviews'] = data['Review_Text'].apply(preprocess_text)
print(data[['Review_Text', 'Clean_Reviews']].head())
data.to_csv('clean_reviews.csv', index=False)
data = pd.read_csv('clean_reviews.csv')
clean_reviews = data['Clean_Reviews']

# Vektorisierung
# BoW
print("\nBoW Vektorisierung:\n")
vect_bow = CountVectorizer(ngram_range=(1, 2), max_features=1000)
data_bow = vect_bow.fit_transform(clean_reviews)
data_bow = pd.DataFrame(data_bow.toarray(), columns=vect_bow.get_feature_names_out())
print(data_bow)

# Tfidf
print("\nTF-IDF Vektorisierung:\n")
vect_tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
data_tfidf = vect_tfidf.fit_transform(clean_reviews)
data_tfidf = pd.DataFrame(data_tfidf.toarray(), columns=vect_tfidf.get_feature_names_out())
print(data_tfidf)

# Themenmodellierung
# LDA
lda = LatentDirichletAllocation(n_components=10,learning_method='online',random_state=42,max_iter=1)
lda_data = lda.fit_transform(data_tfidf)
lda_topic = lda_data[0]
n_words = 10
words = vect_tfidf.get_feature_names_out()

print("\nTop Words for each LDA Topic:")
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx + 1}: " + ", ".join([words[i] for i in topic.argsort()[-n_words:]]))

print("\nReview 1 - LDA Topics: ")
for i, topic in enumerate(lda_topic):
    print(f"Topic {i + 1}: {topic * 100:.2f}")

# LSA
lsa = TruncatedSVD(n_components=6, algorithm='randomized', n_iter=10,random_state=42)
lsa_data = lsa.fit_transform(data_tfidf)
lsa_topic = lsa_data[0]
n_words = 10
words = vect_tfidf.get_feature_names_out()

print("\nTop Words for each LSA Topic:")
for topic_idx, topic in enumerate(lsa.components_):
    print(f"Topic {topic_idx + 1}: " + ", ".join([words[i] for i in topic.argsort()[-n_words:]]))

print("\nReview 1 - LSA Topics: ")
for i, topic in enumerate(lsa_topic):
    print(f"Topic {i + 1}: {topic * 100:.2f}")

# Wordcloud





