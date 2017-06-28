from pandas_datareader import data, wb
import pandas_datareader as pdr
import datetime
import requests_cache


class StockDataSource:
    
    def __init__(self, expiration_time_delta=datetime.timedelta(minutes=60)):
        self.session = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after=expiration_time_delta)
    
    def get_ticker(self, symbol, start_date, end_date):
        raise NotImplementedError( "AbstractClassError" )
        

class YahooDataSource(StockDataSource):
    def __init__(self):
        StockDataSource.__init__(self)
    
    def get_tickers(self, symbol, start_date, end_date):

        panel = data.DataReader(name=symbol, data_source='yahoo', start=start_date, end=end_date, session=self.session)
        df = panel.to_frame().unstack(level=1)
        df.columns = df.columns.swaplevel(0,1)
        return df
    
    def get_ticker(self, symbol, start_date, end_date):
        return data.DataReader(name=symbol, data_source='yahoo', start=start_date, end=end_date, session=self.session)