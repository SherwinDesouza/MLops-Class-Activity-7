import pandas as pd
import numpy as np
import os

def processing():
    df = pd.read_csv(r'./raw_data.csv')

    # Normalize the data
    df["temperature"] = np.abs((df["temperature"] - df["temperature"].mean()) / df["temperature"].std())
    df["wind_speed"] = np.abs((df["wind_speed"] - df["wind_speed"].mean()) / df["wind_speed"].std())
    df["humidity"] = np.abs((df["humidity"] - df["humidity"].mean()) / df["humidity"].std())

    # Save the processed data
    df.to_csv('processed_data.csv', index=False)


processing()    
