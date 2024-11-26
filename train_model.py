import requests
import pandas as pd
from datetime import datetime
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import numpy as np


df = pd.read_csv(r'C:\Users\PAX\MLops-Class-Activity-7\processed_data.csv')

X = df[["humidity", "wind_speed"]]
y = df["temperature"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)


with open("model.pkl", "wb") as f:
    pickle.dump(model, f)