import os

import pandas as pd
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


file_path = ".."
class_labels = ['negative', 'positive']
metrics_file = "..\model_metrics.csv"


def get_sentiments():
    data = pd.read_csv(f"{file_path}\IMDB_Dataset.csv")
    data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
    X_train, X_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.2,
                                                        random_state=42)
    return y_train, y_test


def import_vectors(prefix_name):
    print("Importando vector de caracteristicas de entrenamiento...")
    file = f"{file_path}\{prefix_name}-train.npz"
    train_vectors = load_npz(file)
    print("Importando vector de caracteristicas de prueba...")
    file = f"{file_path}\{prefix_name}-test.npz"
    test_vectors = load_npz(file)
    return train_vectors, test_vectors


def train_model(model, train_vectors, y_train):
    print(f"Entrenamiento de modelo: {type(model).__name__}")
    model.fit(train_vectors, y_train)
    return model


def save_model(trained_model):
    model_name = type(trained_model).__name__
    print(f"Guardando modelo: {model_name}")
    with open(f"{file_path}\{model_name}.pkl", "wb") as file:
        pickle.dump(trained_model, file)


def evaluate_model(model, X_test, y_test, class_labels):
    print(f"\nEvaluando modelo: {type(model).__name__}")
    y_pred = model.predict(X_test)

    # Métricas detalladas
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    macro_precision = report["macro avg"]["precision"]
    macro_recall = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]

    # Mostrar métricas principales
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Guardar métricas en CSV
    metrics_data = {
        "Model": [type(model).__name__],
        "Accuracy": [acc],
        "Macro Precision": [macro_precision],
        "Macro Recall": [macro_recall],
        "Macro F1-Score": [macro_f1],
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(metrics_file, mode='a', header=not os.path.exists(metrics_file), index=False)

    # Mostrar matriz de confusión con etiquetas
    print("\nMatriz de Confusión:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    print(cm_df)


def main():
    y_train, y_test = get_sentiments()
    train_vectors, test_vectors = import_vectors("tfidf")

    models = [
        LogisticRegression(max_iter=200, random_state=42),
        SVC(kernel='linear', random_state=42),
        RandomForestClassifier(n_estimators=150, random_state=42),
        MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=42)
    ]

    for model in models:
        trained_model = train_model(model, train_vectors, y_train)
        save_model(trained_model)
        evaluate_model(trained_model, test_vectors, y_test, class_labels)


if __name__ == "__main__":
    main()
