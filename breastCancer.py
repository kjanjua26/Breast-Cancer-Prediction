import pandas as pd 
import numpy as np 
import matplotlib
from matplotlib import pyplot as plt 
import sklearn as sl 
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import  train_test_split

# Data Formatting 
data = pd.read_csv("data.csv",header=0)
data = data.drop("id", 1)
data = data.drop("Unnamed: 32",1)
mapping = {'M' : 0, 'B' : 1}
data['diagnosis'] = data['diagnosis'].map(mapping)
features = list(data.columns[1:31]) # Appending all the columns in feature vector
train_X, test_X, train_Y, test_Y = train_test_split(data[features], data['diagnosis'].values, test_size=0.20, random_state=42)

# Model
model = Sequential()
model.add(Dense(input_dim=30, output_dim=30))
model.add(Dense(input_dim=30, output_dim=30))
model.add(Dense(input_dim=30, output_dim=30))
model.add(Dense(input_dim=30, output_dim=2))
model.add(Activation("sigmoid"))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

scaler = StandardScaler()
model.fit(scaler.fit_transform(train_X.values), train_Y, epochs=50) # We get accuracy upto 98%
y_prediction = model.predict_classes(scaler.transform(test_X.values))

print "\nAccuracy" , np.sum(y_prediction == test_Y) / float(len(test_Y))
