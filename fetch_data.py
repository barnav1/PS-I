from meteostat import Hourly, Point
from datetime import datetime
import csv

location = Point(17.511, 78.394)  # Hyderabad


start_time = datetime(2025, 5, 1)
end_time = datetime(2025, 6, 1)

data = Hourly(location, start_time, end_time)
data = data.normalize()
data = data.fetch()

print(data)

data = data[['temp']]
data = data.reset_index()


data.to_csv('temperature_data.csv', index=False)