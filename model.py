import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
dataset=pd.read_csv(r"C:\Users\aayus\Documents\hiring.csv")
dataset['experience'].fillna(0, inplace=True)
dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)
X=dataset.iloc[:,:3]
def convert_to_int(word):
    word_dict={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'zero':0,'0':0}
    return word_dict[word];
X['experience']=X['experience'].apply(lambda x : convert_to_int(x))
y=dataset.iloc[:,-1]
#Splititng Training and Test Set
#Since we have a very small dataset,we will train our model with all available data.
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
#Fitting model with training data
regressor.fit(X,y)
#Saving the model to disk
pickle.dump(regressor,open("model1.pk1",'wb'))
