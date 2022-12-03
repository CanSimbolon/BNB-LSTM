import yfinance as yf
stock_symbol = 'bnb-usd'
dataset = yf.download(tickers=stock_symbol,period='max')
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.layers import LSTM
import plotly.graph_objects as go
#Using Tensorflow Backend
values=dataset.values
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)
values = scaled
train_X, test_X, train_y, test_y = train_test_split(values, values[:, 3], test_size=0.2, shuffle=False)
prediction_length=45 # Jumlah hari yang ingin di prediksi
total_train = len(train_X) - (len(train_X) % prediction_length)
train_X = train_X[:total_train]
train_y = train_y[:total_train]
total_test = len(test_X) - (len(test_X) % prediction_length)
test_X = test_X[:total_test] 
test_y = test_y[:total_test]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
from keras.engine import sequential
#Design Network
model=Sequential()
model.add(LSTM(50,input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
history = model.fit(train_X, train_y, epochs=10, batch_size=8, validation_data=(test_X, test_y), verbose=2)
model.save("model.h5")
yhat = model.predict(test_X)
test_X=test_X.reshape((test_X.shape[0],test_X.shape[2])) #1

import plotly.express as px
st.title('PREDIKSI HARGA BNB')
def plot(target_ticks, targets, predictions, title, full_width=True):
    df = pd.DataFrame([target_ticks, targets, predictions]).T
    df.rename(columns={0: 'Waktu', 1: 'Aktual', 2: 'Prediksi'}, inplace=True)
    fig = px.line(df,
        x=df['Waktu'],
        y=[df['Aktual'], df['Prediksi']],
        title=title)
    st.plotly_chart(fig, use_container_width=full_width)

def plot_prediction_v2(past_data_date, prediction_date, past_data, predictions, full_width=True):
        past_dataframe = pd.DataFrame([past_data_date, past_data]).T
        predicted_dataframe = pd.DataFrame([prediction_date, predictions]).T

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=past_dataframe[0],
            y=past_dataframe[1],
            name="Data histori"
        ))
        fig.add_trace(go.Scatter(
            x=predicted_dataframe[0],
            y=predicted_dataframe[1],
            name="Prediksi"
        ))
        st.plotly_chart(fig, use_container_width=full_width)
inv_yhat=concatenate((yhat,test_X[:,1:]),axis=1)
inv_yhat=scaler.inverse_transform(inv_yhat)
inv_yhat=inv_yhat[:,0]
#invert scaling for actual
test_y=test_y.reshape((len(test_y),1))
inv_y=concatenate((test_y,test_X[:,1:]),axis=1)
inv_y=scaler.inverse_transform(inv_y)
inv_y=inv_y[:,0]
#
pred1 = values.reshape((values.shape[0], 1, values.shape[1])) 
inputpred=pred1[-prediction_length:,:]
prediksinya=model.predict(inputpred)
prediksi_nya=concatenate((prediksinya,test_X[:prediction_length,1:]),axis=1)
prediksi_nya=scaler.inverse_transform(prediksi_nya)
prediksi_nya=prediksi_nya[:,0]
# PLOT 1
testingdate=dataset.index[len(train_X)-1:]
inv_yhat=inv_yhat.flatten()
#inv_yhat=np.append(inv_yhat,prediksi_nya)
plot(testingdate,inv_y,inv_yhat.flatten(),"PREDIKSI")
# PLOT 2
def plot_prediction_v2(past_data_date, prediction_date, past_data, predictions, full_width=True):
        past_dataframe = pd.DataFrame([past_data_date, past_data]).T
        predicted_dataframe = pd.DataFrame([prediction_date, predictions]).T
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=past_dataframe[0],
            y=past_dataframe[1],
            name="Data histori"))
        fig.add_trace(go.Scatter(
            x=predicted_dataframe[0],
            y=predicted_dataframe[1],
            name="Prediksi"))
        st.plotly_chart(fig, use_container_width=full_width)
#Gabungin Indext Terakhir
last_index=dataset.iloc[-1:].index
pred_length=prediction_length
predicted_date = [last_index[0] + pd.Timedelta(days=day + 1) for day in range(pred_length)]
title = f"Grafik prediksi {pred_length} hari"
last_data = values.reshape((values.shape[0], 1, values.shape[1])) 
last_data =last_data[-prediction_length:]
predicted = model.predict(last_data)
print(test_X.shape,train_X.shape)
past_data=dataset.Close.iloc[-360:]
past_data_date = past_data.index
predicted=concatenate((predicted,test_X[:prediction_length,1:]),axis=1)
predicted=scaler.inverse_transform(predicted)
predicted=predicted[:,0]
plot_prediction_v2(past_data_date, predicted_date, past_data, predicted)
print(predicted)
print(dataset.Close[-1])
print(predicted[-1])
print(predicted[1])
#Tabel
st.write("Tabel Investasi Binance Coin")
a = st.number_input("Masukkkan jumlah uang investasi")
b =((predicted[-1]-dataset.Close[-1])/predicted[-1])*100
c = a+(a*(b/100))
binance=[[a,"Uang Investasi"],[b,"Presentase Perubahan Harga"],[c,"Imbal Balik Investasi"]]
tabelbnb = pd.DataFrame(binance, columns=['Nominal', 'Keterangan'])
st.table(tabelbnb)
#python -m streamlit run gui1.py