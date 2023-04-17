import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# On va chercher notre dataset
dataFrame = pd.read_excel("SOLAR_CELLS_IA_DATASET.xlsx")
print(dataFrame.head())

# On supprime les colonnes non nécessaire au traitement.
dataFrame.drop(['Numéro '], axis=1, inplace=True)
dataFrame.drop(['Donnor'], axis=1, inplace=True)
dataFrame.drop(['Acceptor'], axis=1, inplace=True)
dataFrame.drop(['Référence'], axis=1, inplace=True)

print(dataFrame.head())

# On met toutes les valeurs des PCE à 0 si inférieur à 10 sinon à 1
dataFrame.PCE[dataFrame.PCE < 10.0] = 0
dataFrame.PCE[dataFrame.PCE >= 10.0] = 1

print(dataFrame.head())


# Y correspond aux résultats attendus 
Y = dataFrame["PCE"].values
Y = Y.astype('int')

# X correspond aux features de notre Dataset
X = dataFrame.drop(labels=["PCE"], axis=1)

# On split nos données avec test_size le pourcentage de valeur utilisée comme valeur de test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)


from sklearn.ensemble import RandomForestClassifier

# On choisi notre model, ici RandomForest
model = RandomForestClassifier(n_estimators = 10, random_state = 40)

# On entraine notre model
model.fit(X_train, y_train)

# On essaie de prédire les valeurs de test
prediction_test = model.predict(X_test)

# On caclul la précision du model
from sklearn import metrics

print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))

# On affiche l'importance de chaque features
feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)
