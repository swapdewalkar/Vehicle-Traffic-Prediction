import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
data = pd.read_csv('C:\\train.csv', parse_dates=['DateTime'], index_col='DateTime',date_parser=dateparse)
data.head()
data_j1 = data.loc[data['Junction'] == 1]
data_j1 = data_j1['Vehicles']
#plt.plot(data_j1.loc[:,['Vehicles']])

#test_stationarity(data_j1)

j1_log=np.log(data_j1)
rolling_avg = pd.rolling_mean(j1_log,10)
#plt.plot(rolling_avg)
j1 = j1_log - rolling_avg
#print(j1.head())
j1.dropna(inplace = True)
decomposition = seasonal_decompose(j1_log)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
j1 = ts_log_decompose
#plt.subplot(411)
#plt.plot(j1_log, label='Original')
#plt.legend(loc='best')
#plt.subplot(412)
#plt.plot(trend, label='Trend')
#plt.legend(loc='best')
#plt.subplot(413)
#plt.plot(seasonal,label='Seasonality')
#plt.legend(loc='best')
#plt.subplot(414)
#plt.plot(residual, label='Residuals')
#plt.legend(loc='best')
#plt.tight_layout()

#test_stationarity(j1)
#plt.plot(test)


lag_acf = acf(j1, nlags=20)
lag_pacf = pacf(j1, nlags=20, method='ols')
#Plot ACF: 
#plt.axhline(y=0,linestyle='--',color='gray')
#plt.axhline(y=-1.96/np.sqrt(len(j1)),linestyle='--',color=
#plt.subplot(121) 
#plt.plot(lag_acf)'gray')
#plt.axhline(y=1.96/np.sqrt(len(j1)),linestyle='--',color='gray')
#plt.title('Autocorrelation Function')
##Plot PACF:
#plt.subplot(122)
#plt.plot(lag_pacf)
#plt.axhline(y=0,linestyle='--',color='gray')
#plt.axhline(y=-1.96/np.sqrt(len(j1)),linestyle='--',color='gray')
#plt.axhline(y=1.96/np.sqrt(len(j1)),linestyle='--',color='gray')
#plt.title('Partial Autocorrelation Function')
#plt.tight_layout()
p = 0 
q = 0
for i in range (0,len(lag_acf)):
    if lag_acf[i]<1.96/np.sqrt(len(j1)):
        p=i
        break
for i in range (0,len(lag_acf)):
    if lag_pacf[i]<1.96/np.sqrt(len(j1)):
        q=i
        break
print(p,q)
p = int(p)
q = int(q)
model = ARIMA(j1_log, order=(p, 0, q))  
results_AR = model.fit(disp=-1)  
model = ARIMA(j1_log, order=(p, 0, q))  
results_MA = model.fit(disp=-1)  
model = ARIMA(j1_log, order=(p, 0, q))  
results_ARIMA = model.fit(disp=-1)  
#plt.plot(j1)
#plt.plot(results_ARIMA.fittedvalues, color='red')
#plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-j1)**2))
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print (predictions_ARIMA_diff.head())
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print (predictions_ARIMA_diff_cumsum.head())
predictions_ARIMA_log = pd.Series(j1_log.ix[0], index=j1_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(data_j1)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-data_j1)**2)/len(data_j1)))