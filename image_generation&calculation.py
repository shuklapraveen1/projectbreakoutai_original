import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sb
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

#File Reader
stock=pd.read_csv("C:/Users/shukl/breakoutaiproject/file_uploaders/Tesla.csv")

# File path for saving the plot
save_path = "C:/Users/shukl/breakoutaiproject/graph/"

# Check if the directory exists, and create it if not
if not os.path.exists(save_path):
    os.makedirs(save_path)


#Line Plot
plt.figure(figsize=(20,10))
plt.plot(stock['Close'])
plt.title('Line Plot.', fontsize=10)
plt.ylabel('Price in dollars.')
plt.savefig(os.path.join(save_path, 'Line_Plot.png'), dpi=300, bbox_inches='tight')

#Distplot

stock = stock.drop(['Adj Close'], axis=1)

features = ['Open', 'High', 'Low', 'Close', 'Volume']

plt.subplots(figsize=(20,10))
plt.title('Stock Price - Distplot')

for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.distplot(stock[col])

plt.savefig(os.path.join(save_path, 'Displot.png'), dpi=300, bbox_inches='tight')


#Box plot
plt.subplots(figsize=(20,10))
plt.title('Stock Price - Box Plot')
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.boxplot(stock[col])
  
plt.savefig(os.path.join(save_path, 'Boxplot.png'), dpi=300, bbox_inches='tight')


#Histogram
splitted = stock['Date'].str.split('/', expand=True)

stock['day'] = splitted[1].astype('int')
stock['month'] = splitted[0].astype('int')
stock['year'] = splitted[2].astype('int')

stock['is_quarter_end'] = np.where(stock['month']%3==0,1,0)

data_grouped = stock.drop('Date', axis=1).groupby('year').mean()
plt.subplots(figsize=(20,10))
plt.title('Stock Price Histograms')
for i, col in enumerate(['Open', 'High', 'Low', 'Close','Volume']):
  plt.subplot(2,3,i+1)
  data_grouped[col].plot.bar()

plt.savefig(os.path.join(save_path, 'Histograms.png'), dpi=300, bbox_inches='tight')


#Pie Chart
stock['open-close'] = stock['Open'] - stock['Close']
stock['low-high'] = stock['Low'] - stock['High']
stock['target'] = np.where(stock['Close'].shift(-1) > stock['Close'], 1, 0)
plt.figure(figsize=(8, 8))
plt.pie(stock['target'].value_counts().values, labels=[0, 1], autopct='%1.1f%%')
plt.title('Stock Price - Pie Chart')
pie_chart_path = os.path.join(save_path, 'Piechart.png')
plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')



#scatter matrix

plt.figure(figsize=(10, 8))
sb.pairplot(stock[['Open', 'Close', 'High', 'Low', 'Volume']])
plt.savefig(os.path.join(save_path, 'Scattermatrix.png'), dpi=300, bbox_inches='tight')

#HeatMap
plt.figure(figsize=(10, 10))
sb.heatmap(stock.drop('Date', axis=1).corr() > 0.9, annot=True, cbar=False)
plt.title('Stock Price - Heat Map')
plt.savefig(os.path.join(save_path, 'Heatmap.png'), dpi=300, bbox_inches='tight')

# Convert 'Date' column to datetime
stock['Date'] = pd.to_datetime(stock['Date'])
plt.figure(figsize=(12, 6))
# Loop through each row to plot candlesticks
for idx, row in stock.iterrows():
    color = 'green' if row['Close'] >= row['Open'] else 'red'
    # Plot the high-low line
    plt.plot([row['Date'], row['Date']], [row['Low'], row['High']], color=color, linewidth=1.5)
    # Plot the open-close rectangle (thick line)
    plt.plot([row['Date'], row['Date']], [row['Open'], row['Close']], color=color, linewidth=6)
plt.title('Stock Price - Candlestick Chart', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price in dollars', fontsize=14)
plt.xticks(rotation=45)
plt.savefig(os.path.join(save_path, 'Candlestick.png'), dpi=300, bbox_inches='tight')

