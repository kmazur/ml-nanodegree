from datetime import datetime, timedelta
from traceback import print_last

import pandas as pd
import numpy as np


from StockDataSource import YahooDataSource
from StockPricePredictor import StockPricePredictor
import Utils

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

def predictARIMA(series, prediction_days):
        log = series.copy()
        log = np.log(log)
        log_diff = log - log.shift(1)
        log.replace([np.inf, -np.inf], np.nan, inplace=True)
        log.dropna(inplace=True)

        (trend, sesonal, residual) = Utils.decompose(log, freq=prediction_days)
        res = residual.dropna()

        res2 = pd.Series(data=log.values.squeeze(), index=log.index)
        res3 = res2.reset_index(drop=True).values
        arima = {}
        for p in range(1, 3):
            for q in range(1, 3):
                try:
                    arima = ARIMA(res3, order=(p, 1, q)).fit()
                    break
                except:
                    pass
        results = arima.predict(start=len(res3)-1, end=len(res3)+prediction_days, dynamic=True)

        predictions_ARIMA_diff = pd.Series(results, copy=True)
        predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

        start = log.index[-1]
        end = start + pd.tseries.offsets.DateOffset(days=2*prediction_days)
        us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        index = pd.DatetimeIndex(start=start,end=end, freq=us_bd)

        preds = predictions_ARIMA_diff_cumsum[-prediction_days:].values
        dateIndex = index.values[0:prediction_days]
        predictions_ARIMA_diff_date = pd.Series(data=preds, index=dateIndex)

        lastValuePred = pd.Series(log.ix[-1], index=dateIndex)
        predicted = lastValuePred.add(predictions_ARIMA_diff_date, fill_value=0)

        predictions = np.exp(predicted)
        return predictions
    
    
    
    