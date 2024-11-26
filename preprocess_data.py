import pandas as pd
import numpy as np
df = pd.read_csv(r'C:\Users\PAX\MLops-Class-Activity-7\raw_data.csv')

df["temperature"] = np.abs((df["temperature"] - df["temperature"].mean()) / df["temperature"].std())
df["wind_speed"] = np.abs((df["wind_speed"] - df["wind_speed"].mean()) / df["wind_speed"].std())
df["humidity"] = np.abs((df["humidity"] - df["humidity"].mean()) / df["humidity"].std())


df.to_csv("processed_data.csv", index=False)