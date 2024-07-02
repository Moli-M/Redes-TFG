import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras import layers
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

archivos_csv = ['ExternalServer\\CIDDS-001-external-week1.csv'#, 'ExternalServer\\CIDDS-001-external-week2.csv', 
                #'ExternalServer\\CIDDS-001-external-week3.csv', 
                #'ExternalServer\\CIDDS-001-external-week4.csv'
                #'OpenStack\\CIDDS-001-internal-week1.csv', 'OpenStack\\CIDDS-001-internal-week2.csv', 
                #'OpenStack\\CIDDS-001-internal-week3.csv', 'OpenStack\\CIDDS-001-internal-week4.csv',
                ]
ruta = "C:\\Users\\diego\\Desktop\\TFG\\CIDDS-001\\traffic\\"

dataframes = []

for archivo in archivos_csv:
    df = pd.read_csv(ruta+archivo)
    dataframes.append(df)

dataframe_final = pd.concat(dataframes, axis=0)

X = dataframe_final.drop('class', axis=1)
y = dataframe_final['class']

le = LabelEncoder()
for col in X.select_dtypes(include=['object']):
    X[col] = le.fit_transform(X[col])

scaler = StandardScaler()
X[X.select_dtypes(include=['float64']).columns] = scaler.fit_transform(X.select_dtypes(include=['float64']))

y = le.fit_transform(y)

onehot_encoder = OneHotEncoder()

y = keras.utils.to_categorical(y, num_classes=5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

max_features = int(np.sqrt(X_train.shape[1]))
model = RandomForestClassifier(n_estimators=10000, 
                                max_depth=50, 
                                min_samples_split=10,                       
                                min_samples_leaf=5, 
                                max_features=max_features, 
                                class_weight='balanced', 
                                criterion='gini'
                            )

history = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

model.save('../Modelos/random_forest.keras')
