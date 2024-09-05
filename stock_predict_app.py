import yfinance as yf
import streamlit as st
import pandas as pd

from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go

START = '2018-01-01'
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock prediction App")
stocks = ("AAPL","GOOG","MSFT","TSLA")
selected_stocks = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("years of predcition:",1,5)
n_month = st.slider("month of predcition:",1,12)
period = n_years * 365
period_m = n_month * 30

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text("Load data..")
data = load_data(selected_stocks)
data_load_state.text("Loading data...done!")

st.subheader('Raw data')
st.write(data.tail())

def plot_row_data(): 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_row_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
#years
future_y = m.make_future_dataframe(periods=period)
forecast_y = m.predict(future_y)
#month
future_m = m.make_future_dataframe(periods=period_m)
forecast_m = m.predict(future_m)


# Show and plot forecast
st.subheader('Forecast data Year')
st.write(forecast_y.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast_y)
st.plotly_chart(fig1)

st.subheader('Forecast data Month')
st.write(forecast_m.tail())

st.write(f'Forecast plot for {n_month} month')
fig1 = plot_plotly(m, forecast_m)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast_y)
st.write(fig2)
