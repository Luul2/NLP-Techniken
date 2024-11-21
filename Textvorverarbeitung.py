import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

datafile = r"C:\Users\Maik und Luisa\Downloads\archive\DisneylandReviews.csv"
data = pd.read_csv(datafile, encoding='ISO-8859-1')
print(data.info())
reviews = data['Review_Text']

