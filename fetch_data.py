from meteostat import Hourly, Point
from datetime import datetime

location = Point(17.511, 78.394)  # Hyderabad


start_time = datetime(2025, 1, 1)
end_time = datetime(2025, 6, 1)

data = Hourly(location, start_time, end_time)
data = data.fetch()


data = data[["temp"]]
data = data.reset_index()


data.to_csv("temperature_data.csv", index=False)
