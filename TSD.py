import csv
import pandas as pd
import numpy as np
df=pd.read_csv('train.csv')
print df.head()
grp=df.groupby(['Junction'])
group3=grp.get_group(3);
print group3
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
data = pd.read_csv('train.csv', parse_dates=['DateTime'], index_col='DateTime',date_parser=dateparse)
ts = data['Vehicles'] 
ts_train=ts
print ts.head()
plt.plot(ts)
plt.show()
print "swapnil"
from statsmodels.tsa.arima_model import ARIMA
print "swapnil"
X=ts_train.values
size = int(len(X) * 0.75)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print yhat, obs
error = mean_squared_error(test, predictions)

