import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import seaborn as sns
from keras.models import load_model
import streamlit as st
from yahoofinancials import YahooFinancials
from datetime import datetime
import yfinance as yf


startdate = datetime(2010, 1, 29)
enddate= datetime.today()

st.title('Stock Trend Prediction')

user_input=st.text_input('Enter Stock Ticker', 'AAPL')
df=yf.download(user_input ,start=startdate, end=enddate)

#Describing Data
st.subheader('Data from 2010-2024')
st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

# !00 days moving average
st.subheader('Closing Price vs Time chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
st.pyplot(fig)

#200 days moving average
st.subheader('Closing Price vs Time chart with 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
st.pyplot(fig)

# Splitting Data into training and testing
train_data=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
test_data=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

# Scaling train data
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
data_training=scaler.fit_transform(train_data)


#Loading model
model=load_model('Apple_Stock.h5')

# testing part
past_100_days=train_data.tail(100)
final_df=np.concatenate((past_100_days, test_data), axis=0)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, :])
x_test, y_test = np.array(x_test),np.array(y_test)

y_predicted=model.predict(x_test)
y_predicted_final = y_predicted[:, -1, :]

# Visualization Actual vs Predicted.
st.subheader('Predicted vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted_final,'r',label='Predicted Price')
plt.title('Actual vs Predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


    
