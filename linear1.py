import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# input data here
df=pd.read_csv('dataset name')

scalar = MinMaxScaler()
scalar.fit(df)
df= scalar.transform(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=500, verbose=0)


# model's evaluation
model.evaluate(X_test, y_test, verbose=1, steps= 10)

# making predictions!
predictions = model.predict()# input the data that you want

