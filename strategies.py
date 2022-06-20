import numpy as np
from ta.trend import ema_indicator, sma_indicator
from ta.momentum import rsi


class TrendFollowingStrategy(object):
    def __init__(self, settings):
        self.settings = {
            'contract_type': 'both', 
            'risk_factor': 0.001,
            'fast_window': 40,
            'slow_window': 80,
            'std_window': 40,
            'breakout_window': 50,
            'exit_multiplier': 3
        }
        if type(settings) == dict:
            if all(key in settings for key in self.settings.keys()):
                self.settings = settings
        self.long_on = True if self.settings['contract_type'] in ['long', 'both'] else False
        self.short_on = True if self.settings['contract_type'] in ['short', 'both'] else False
        self.risk_factor = self.settings['risk_factor']
        self.fast_window = self.settings['fast_window']
        self.slow_window = self.settings['slow_window']
        self.std_window = self.settings['std_window']
        self.breakout_window = self.settings['breakout_window']
        self.exit_multiplier = self.settings['exit_multiplier']
        self.required_data_length = 3 * np.max([
            self.fast_window,
            self.slow_window,
            self.std_window,
            self.breakout_window
        ])
        return

    def __str__(self):
        description = "Future Trend Following Strategy"
        description += "\nintroduced by Andreas Clenow in his Trading Evolved book"
        return description

    def position_size(self, ohlc, portfolio_value):
        pct_changes = (ohlc['close_price'] - ohlc['close_price'].shift(1)) / ohlc['close_price'].shift(1)
        std_dev = np.std(pct_changes.values[-self.std_window:])
        size = self.risk_factor * portfolio_value / std_dev
        return size

    def compute_indicators(self, ohlc):
        if ohlc.shape[0] < self.required_data_length:
            raise Exception("Length of passed OHLC data too small")
        ohlc['ema_fast'] = ema_indicator(ohlc['close_price'], self.fast_window)
        ohlc['ema_slow'] = ema_indicator(ohlc['close_price'], self.slow_window)
        ohlc['uptrend'] = np.where(ohlc['ema_fast'] >= ohlc['ema_slow'], True, False)
        ohlc['downtrend'] = np.where(ohlc['ema_fast'] < ohlc['ema_slow'], True, False)
        return ohlc

    def open_long(self, ohlc):
        long_signal = ohlc['uptrend'].iloc[-1] and ohlc['close_price'].iloc[-1] == np.max(ohlc['close_price'].iloc[-self.breakout_window:])
        long_signal = long_signal and self.long_on
        return long_signal

    def open_short(self, ohlc):
        short_signal = ohlc['downtrend'].iloc[-1] and ohlc['close_price'].iloc[-1] == np.min(ohlc['close_price'].iloc[-self.breakout_window:])
        short_signal = short_signal and self.short_on
        return short_signal

    def close_position(self, ohlc, position):
        if position.status != 'open':
            raise Exception('Unable to close a closed position.')
        price = ohlc['close_price'].iloc[-1]
        std_dev = ohlc['close_price'].diff().iloc[-self.std_window:].std()
        if position.drawdown >= self.exit_multiplier*std_dev:
            return True
        if (position.side == 'long') and (not ohlc['uptrend'].iloc[-1]):
            return True
        if (position.side == 'short') and (not ohlc['downtrend'].iloc[-1]):
            return True
        return False



class DemoStrategy():
    def __init__(self, risk_factor):
        self.risk_factor = risk_factor
        self.required_data_length = 100
        return

    def __str__(self):
        description = "Demo EMA Strategy"
        description += "\n Buy when RSI crossover 30 within an uptrend;"
        description += "\n Sell when RSI crossunder 70 within an uptrend;"
        return description

    def position_size(self, ohlc, portfolio_value):
        pct_changes = (ohlc['close_price'] - ohlc['close_price'].shift(1)) / ohlc['close_price'].shift(1)
        std_dev = np.std(pct_changes.values[-20:])
        size = self.risk_factor * portfolio_value / (10*std_dev)
        return size

    def compute_indicators(self, ohlc):
        if ohlc.shape[0] < self.required_data_length:
            raise Exception("Length of passed OHLC data too small")
        ohlc['rsi'] = rsi(ohlc['close_price'], 3)
        ohlc['ema_fast'] = ema_indicator(ohlc['close_price'], 10)
        ohlc['ema_slow'] = ema_indicator(ohlc['close_price'], 20)
        ohlc['uptrend'] = np.where(ohlc['ema_fast'] >= ohlc['ema_slow'], True, False)
        ohlc['downtrend'] = np.where(ohlc['ema_fast'] < ohlc['ema_slow'], True, False)
        return ohlc

    def open_long(self, ohlc):
        long_signal = (ohlc['uptrend'].iloc[-1] and not ohlc['uptrend'].iloc[-2])
        return long_signal

    def open_short(self, ohlc):
        short_signal = (ohlc['downtrend'].iloc[-1] and not ohlc['downtrend'].iloc[-2])
        return short_signal

    def close_position(self, ohlc, position):
        if position.status != 'open':
            raise Exception('Unable to close a closed position.')
        if (position.side == 'long') and (not ohlc['uptrend'].iloc[-1]):
            return True
        if (position.side == 'short') and (not ohlc['downtrend'].iloc[-1]):
            return True
        return False