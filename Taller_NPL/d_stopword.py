import nltk
import gensim.downloader as api

# Descarga el conjunto de datos de stopwords
nltk.download('stopwords')
word2vec = api.load("word2vec-google-news-300")