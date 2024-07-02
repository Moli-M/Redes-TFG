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

archivos_csv = [#'ExternalServer\\CIDDS-001-external-week1.csv'#, 'ExternalServer\\CIDDS-001-external-week2.csv', 
                #'ExternalServer\\CIDDS-001-external-week3.csv', 
                #'ExternalServer\\CIDDS-001-external-week4.csv',
                'OpenStack\\CIDDS-001-internal-week1.csv', #'OpenStack\\CIDDS-001-internal-week2.csv', 
                #'OpenStack\\CIDDS-001-internal-week3.csv', 'OpenStack\\CIDDS-001-internal-week4.csv',
                ]
ruta = "..\\CIDDS-001\\traffic\\"

dataframes = []

for archivo in archivos_csv:
    df = pd.read_csv(ruta+archivo)
    dataframes.append(df)

dataframe_final = pd.concat(dataframes, axis=0)

X = dataframe_final.drop('class', axis=1)
y = dataframe_final['class']

le = LabelEncoder()
for col in X.select_dtypes(include=['object']):
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])

scaler = StandardScaler()
X[X.select_dtypes(include=['float64']).columns] = scaler.fit_transform(X.select_dtypes(include=['float64']))

y = le.fit_transform(y) 

onehot_encoder = OneHotEncoder()

y = keras.utils.to_categorical(y, num_classes=5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = keras.Sequential([
    layers.LSTM(10, return_sequences=True, input_shape=(X.shape[1], 1)),
    layers.LSTM(100, return_sequences=True, activation='tanh'),
    layers.Dropout(0.2),
    layers.LSTM(100, activation='tanh'),
    layers.Dropout(0.2),
    layers.Dense(5, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train
                    , epochs=50, batch_size=250, validation_split=0.2
                    )


print("Accuracy de entrenamiento:", history.history['accuracy'])

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

def get_modelo():
    
    return model


modelo_wrapped = KerasClassifier(build_fn=get_modelo, epochs=1, batch_size=50)

print("\n\nVALIDACION CRUZADA\n\n")

kf = KFold(n_splits=5)


 
scores = cross_val_score(modelo_wrapped, X_train, y_train, cv=kf, scoring="accuracy")
 
print("Metricas cross_validation", scores)
 
print("Media de cross_validation", scores.mean())
 
preds = model.predict(X_test)
y_pred = np.argmax(preds, axis = 1)

lee = LabelEncoder()


y_test_cat = np.argmax(y_test, axis = 1)
score_pred = metrics.accuracy_score(y_test_cat, y_pred)
 
print("Metrica en Test", score_pred)

model.save('../Modelos/modelo_lstm.keras')