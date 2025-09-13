import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Escolhi o dataset de câncer de mama porque é um clássico para classificação binária,
# com features numéricas e um target binário (maligno/benigno).

cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

 
X = df.drop('target', axis=1)
y = df['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=5000)
}

results = {}

for name, model in models.items():

    print(f"Treinando {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=cancer.target_names, output_dict=True)

    results[name] = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"]
    }

    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cancer.target_names, yticklabels=cancer.target_names)
    plt.title(f"Matriz de Confusão - {name}")
    plt.xlabel("Predição")
    plt.ylabel("Real")
    plt.show()


for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"Acurácia: {metrics['accuracy']:.2f}")
    print(f"Precisão: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"F1-Score: {metrics['f1_score']:.2f}")



plt.figure(figsize=(6,4))
plt.bar(results.keys(), [m["accuracy"] for m in results.values()], color=["skyblue","orange","green"])

plt.title("Comparação de Acurácia dos Modelos")
plt.ylabel("Acurácia")
plt.show()



# CONLUSÃO:

# ---> Qual modelo se saiu melhor?

# O modelo que se saiu melhor foi a Regressão Logística, com acurácia de 0.98. '
# O KNN também foi bem, com 0.96, e a Decision Tree teve 0.93, mas ainda assim não ficou muito atrás.

# ---> O resultado faz sentido?

# O resultado faz sentido, porque os dados são limpos, numéricos e de classificação binária, 
# então a Regressão Logística consegue separar bem os casos.


# ---> O que dá pra melhorar?

# Normalizar os dados (o KNN pode melhorar com isso).
# Testar outros hiperparâmetros nos modelos.
# Adicionar mais variáveis, se existissem, pra capturar padrões mais complexos.
