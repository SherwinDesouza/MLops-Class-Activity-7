import requests
import pandas as pd
from datetime import datetime
import time


from datetime import datetime

API_KEY = "API_KEY" # Add your API key here
CITY = "City Name" # Add your city name here
URL = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
weather_data_list = []
def fetch_weather_data():
    count =0
    while count<500:
        response = requests.get(URL)
        if response.status_code == 200:
            data = response.json()
            weather_data = {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"],
                "weather_condition": data["weather"][0]["description"],
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
            }
            weather_data_list.append(weather_data)
            df = pd.DataFrame(weather_data_list)
            df.to_csv("raw_data.csv", index=False)
            #print(f"Data collected: {weather_data}")  

            #df = pd.DataFrame(weather_data_list)
            #df.to_csv("weather_data.csv", index=False)
        else:
            print(f"Error: {response.status_code}")
        
        #time.sleep(120)
        count+=1
        #print("Data count: ",count)

fetch_weather_data()
