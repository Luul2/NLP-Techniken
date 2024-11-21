import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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


# Vektorisierung mit BoW und Tfidf

vect_bow = CountVectorizer(ngram_range=(1, 2), max_features=1000)
data_bow = vect_bow.fit_transform(clean_reviews)
data_bow = pd.DataFrame(data_bow.toarray(), columns=vect_bow.get_feature_names_out())
print(data_bow)

vect_tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
data_tfidf = vect_tfidf.fit_transform(clean_reviews)
data_tfidf = pd.DataFrame(data_tfidf.toarray(), columns=vect_tfidf.get_feature_names_out())
print(data_tfidf)

