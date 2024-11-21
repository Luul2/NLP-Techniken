import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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

