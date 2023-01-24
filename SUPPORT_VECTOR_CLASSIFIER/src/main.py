import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sns; sns.set(font_scale=1.2)
csv = pd.read_csv('SOLAR_CELLS_IA_DATASET.csv', sep = ';')

features = csv[['Jsc', 'Voc', 'FF']].to_numpy()
label = np.where(csv['PCE'] > 11, 0, 1)

model = svm.SVC(kernel='linear')
model.fit(features, label)


def accuracy(Jsc, Voc,FF):
    if(model.predict([[Jsc, Voc, FF]]))>10:
        print('good accuracy')
    else:
        print('bad accuracy')


accuracy(18.8,0.94,0.69) 
