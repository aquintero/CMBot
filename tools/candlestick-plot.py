import plotly.graph_objects as go
import pandas as pandas
from datetime import datetime

CSV_PATH = '~/path/to/your/data/folder/btcusd.csv'

data = pandas.read_csv(CSV_PATH)

# Go as high as you dare but my computer doesn't like more than 15k
data = data.tail(5000)

# The dataset uses utc timestamps with milliseconds for some reason, so fix that shit
data['time'] = data['time'].apply(lambda utc_ms: datetime.utcfromtimestamp(
    int(utc_ms / 1000)).strftime('%Y-%m-%d %H:%M:%S'))

plot = go.Figure(data=[go.Candlestick(
    x=data['time'], open=data['open'], high=data['high'], low=data['low'], close=data['close'])])

plot.show()
