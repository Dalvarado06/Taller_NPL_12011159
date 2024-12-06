import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

file_path = ".."
stop_words = set(stopwords.words('english'))


def get_dataset():
    data = pd.read_csv(f"{file_path}/IMDB_dataset.csv")
    return data


def dataset_info(data):
    print("Filas iniciales del dataset:")
    print(data.head())

    print("Distribucion de clases:")
    print(data['sentiment'].value_counts())

    print("Datos Nulos:")
    print(data.isnull().sum())


def sample_dataset(data, sample_size):
    data = data.sample(frac=sample_size, random_state=42).reset_index(drop=True)
    return data

def remove_html_tags(data, columnName):
    print("Extraccion de HTML tags")
    def clean_html(text):

        if(isinstance(text, str)):
            return BeautifulSoup(text, "html.parser").get_text()
        return text

    data[columnName] = data[columnName].apply(clean_html)
    return data


def remove_review_stopwords(review):
    words = review.lower().split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)


def remove_stopwords(data):
    print("Eliminando stopwords del corpus...")
    data['cleaned_review'] = data['review'].apply(remove_review_stopwords)
    print("Documento original vs procesado:")
    print(data[['review', 'cleaned_review']].head())
    return data


def split_dataset(data):
    X_train, X_test, y_train, y_test = train_test_split(data['cleaned_review'], data['sentiment'], test_size=0.3,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def tfidf_vectorizer(X_train, X_test):
    vectorizer = TfidfVectorizer(
        max_features=1000,  # limite de features
        ngram_range=(1, 2),  # Capturar palabras en duo o tripletas
        min_df=2,  # Ignora palabras que aparecen en menos de 2 documentos
        max_df=0.9,  # Ignora las palabras que aparecen el 90% de documentos
        sublinear_tf=True,  # Ayuda con las frecuencias altas de terminos
        use_idf=True,  # Baja el peso de palabras que son recurrentes en el texto
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF shape train: {X_train_tfidf.shape}")
    print(f"TF-IDF shape test: {X_test_tfidf.shape}")

    print("Vector TFIDF en primer documento:")
    print(X_train_tfidf.toarray()[0])
    return X_train_tfidf, X_test_tfidf

def save_vectors(X_train_tfidf, X_test_tfidf, prefix_name):
    file = f"{file_path}\{prefix_name}-train.npz"
    save_npz(file, X_train_tfidf)
    file = f"{file_path}\{prefix_name}-test.npz"
    save_npz(file, X_test_tfidf)
    print("Archivos guardados correctamente")

def main():
    print("Descargando Word2Vec preentrenado...")
    nltk.download('punkt_tab')

    data = get_dataset()
    dataset_info(data)
    data = remove_html_tags(data, columnName='review')
    data = remove_stopwords(data)
    X_train, X_test, y_train, y_test = split_dataset(data)
    X_train_tfidf, X_test_tfidf = tfidf_vectorizer(X_train, X_test)
    save_vectors(X_train_tfidf, X_test_tfidf, "wemb")


if __name__ == "__main__":
    main()
