from meteostat import Hourly, Point
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Set the desired location to Hyderabad
location = Point(17.511, 78.394) 

# Set the start and end dates for the temperature data to be fetched
start_time = datetime(2015, 1, 1)
end_time = datetime(2025, 6, 25)

# Fetch the hourly data with the above parameters
data = Hourly(location, start_time, end_time)
data = data.fetch()


# Save the fetched data to a csv file
data.to_csv("temperature_data.csv", index=True)
