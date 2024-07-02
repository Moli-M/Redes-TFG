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

model = keras.Sequential([
    layers.SimpleRNN(10, return_sequences=True, input_shape=(X.shape[1], 1)),
    layers.SimpleRNN(100, return_sequences=True, activation='tanh'),
    layers.Dropout(0.2),
    layers.SimpleRNN(100, activation='tanh'),
    layers.Dropout(0.2),
    layers.Dense(5, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train
                    , epochs=10, batch_size=250, validation_split=0.2
                    )

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

model.save('../Modelos/modelo_rnn.keras')