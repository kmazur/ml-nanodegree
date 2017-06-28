import math

class Portfolio:
    def __init__(self, ticker_data, allocations, investment_value):
        """ticker_data must be normalized"""
        alloced = ticker_data * allocations
        pos_vals = alloced * investment_value
        self.port_val = pos_vals.sum(axis=1)
        self.daily_rets = self.port_val[1:]
        
        self.cum_ret = (self.port_val[-1]/self.port_val[0]) - 1
        self.avg_daily_ret = self.daily_rets.mean()
        self.std_daily_ret = self.daily_rets.std()
        daily_ref = ((1.0 + 0.1) ** (1.0/252)) - 1
        self.sharpe_ratio = (self.daily_rets - daily_ref).mean() / (self.std_daily_ret)
        self.sharpe_ratio_ann = math.sqrt(252) * self.sharpe_ratio
    def print_stats(self):
        print("Cumulative return: ", self.cum_ret)
        print("Average daily ret: ", self.avg_daily_ret)
        print("Std of daily ret:  ", self.std_daily_ret)
        print("Sharpe ratio:      ", self.sharpe_ratio)
        print("Sharpe ratio ann:  ", self.sharpe_ratio_ann)