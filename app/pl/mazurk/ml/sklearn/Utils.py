import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
 
def calculate_return(adj_close_now, adj_close_before):
    return (adj_close_now - adj_close_before) / adj_close_before;

def intersection(a, b):
    return list(set(a) & set(b))

def get_bollinger_bands(series, window=20):
    deviation2 = 2 * pd.rolling_std(series, window, min_periods=window)
    mean = pd.rolling_mean(series, window=window)
    upper = mean + deviation2
    lower = mean - deviation2
    return pd.DataFrame({'UPPER': upper, 'LOWER': lower}, index=series.index)

def get_daily_returns(df):
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    daily_returns.ix[0, :] = 0
    return daily_returns


def test_stationarity(timeseries, window=252):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window)
    rolstd = pd.rolling_std(timeseries, window)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    
    idx = timeseries.reset_index()
    for i in range(idx.shape[0]/window):
        plt.axvline(idx.ix[i*window]['Date'], color='g', linestyle='--', linewidth=0.5)
    
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput
    
def decompose(series, freq):
    decomposition = seasonal_decompose(series, freq=freq)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    return (trend, seasonal, residual)

def plot_decomposition(series, trend, sesonal, residual):
    plt.subplot(411)
    plt.plot(series, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(sesonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()