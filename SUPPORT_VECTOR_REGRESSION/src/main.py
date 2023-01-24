from sklearn.svm import SVR
from sklearn.datasets import make_regression
import pandas as pd

# f = open("filename.txt")
# f.readline()  # skip the header
# data = np.loadtxt(f)

# # Générer des données de régression aléatoires
# X, y = make_regression(n_samples=100, n_features=1, noise=0.1)

# # Initialiser un modèle SVR avec un noyau radial (RBF)
# model = SVR(kernel='rbf')

# # Entraîner le modèle sur les données d'entraînement
# model.fit(X, y)

# # Utiliser le modèle pour faire des prédictions sur de nouvelles données
# y_pred = model.predict([[32]])
# print(y_pred)

# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split

# # Lire les données à partir d'un fichier CSV

# # Séparer les données en caractéristiques (X) et cibles (y)
# X = data[['feature1', 'feature2', 'feature3']]
# y = data['target']

# # Séparer les données en un ensemble d'entraînement et un ensemble de test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Initialiser un modèle de régression linéaire
# model = LinearRegression()

# # Entraîner le modèle sur les données d'entraînement
# model.fit(X_train, y_train)

# # Evaluer le modèle sur les données de test
# score = model.score(X_test, y_test)
# print(score)

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


data = pd.read_excel('D:\Desktop\cours\AP4\projet_recherche\SOLAR_CELLS_IA\DATASET\SOLAR_CELLS_IA_DATASET.xlsx')


X = data.iloc[:, 3].values
y = data.iloc[:, 4].values
z = data.iloc[:, 5].values

print("voici X", X)
print("voici y", y)
print("voici z", z)

from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, y, z)


# print(clf.predict([[-0.8, -1, 4]]))





# print(data)
