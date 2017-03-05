from StockDataSource import YahooDataSource
import datetime
import Utils
import pandas as pd

class StockPricePredictor:
    
    def __init__(self, data_source=YahooDataSource()):
        self.ds = data_source
    
    def fit(self, ticker_symbols, start_date, end_date):
        """
        @param ticker_symbols - an array of tickers names
        @param start_data - start date for analysis
        @param end_data - end date for analysis
        
        @returns Adjusted Close Price of prediction
        """
        
        # Preconditions
        assert start_date < end_date

        # Store to constrain 'predict' method
        self.end_date = end_date
        self.ticker_symbols = ticker_symbols
        
        df = self._get_adj_close(ticker_symbols, start_date, end_date)
        normalized = self._clean_data(df);
        
        return (df, normalized)
    
    def predict(self, ticker_symbols, dates):
        """
        @param ticker_symbols - an array of tickers names
        @param dates - an array of dates to predict the tickers values
        
        @returns an array of predicted ticker values
        """
        
        # Preconditions
        for date in dates:
            assert date > self.end_date
            
        # Allow only fitted tickers
        assert len(Utils.intersection(ticker_symbols, self.ticker_symbols)) == len(ticker_symbols)
        
        return self.real(ticker_symbols, dates)
    
    def _clean_data(self, df):
        df.dropna()
        # Normalize
        return df / df.ix[0, :]
    
    def _get_adj_close(self, ticker_symbols, start_date, end_date):
        df = self._get_ticker('SPY').ix[start_date:end_date, ['Adj Close']]
        df = pd.DataFrame(df)
        df.rename(columns={'Adj Close': 'SPY_ref'}, inplace=True)

        for ticker in ticker_symbols:
            tickerDf = self._get_ticker(ticker)
            series = tickerDf.ix[start_date:end_date, ['Adj Close']]
            tickerDf = pd.DataFrame(series)
            tickerDf.rename(columns={'Adj Close': ticker}, inplace=True)
            df = df.join(tickerDf)
            
        df.drop('SPY_ref', 1, inplace=True)
        return df
        
    
    def real(self, ticker_symbols, dates):
        """
        @param ticker_symbols - an array of tickers names
        @param dates - an array of dates to predict the tickers values
        
        @returns a dictionary of ticker_symbol => date values
        """
        
        # TODO: use get_tickers
        ret = {}
        for ticker in ticker_symbols:
            df = self._get_ticker(ticker)
            ret[ticker] = df['Adj Close'][dates].values
        return ret
    
    def _get_ticker(self, ticker):
        start = datetime.datetime(1800, 1, 1)
        end = datetime.datetime.today()
        return self.ds.get_ticker(ticker, start, end)