import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time as tm
import os

import strategies
import statistics


class PerformanceTracker():
    def __init__(self):
        root = Path(os.getcwd())
        result_dir_path = root / 'backtest_results'
        if os.path.exists(result_dir_path):
            for file_name in os.listdir(result_dir_path):
                file_path = result_dir_path / file_name
                os.remove(file_path)
            os.rmdir(result_dir_path)
        os.mkdir(result_dir_path)
        self.paths = {
            'portfolio': result_dir_path / 'portfolio_evolution.csv',
            'closed_positions': result_dir_path / 'closed_positions.csv',
            'backtest_results': result_dir_path / 'backtest_summary.txt',
            'portfolio_plot': result_dir_path / 'portfolio_evolution.png',
            'positions_profits_plot': result_dir_path / 'positions_profits.png'
        }
        self.PORTFOLIO = list()
        self.CLOSED_POSITIONS = list()
        self.dataframes_created = False
        return

    def track_portfolio_values(self, backtest):
        """Track portfolio's values"""
        portfolio_value = backtest.available_capital
        realized_portfolio_value = backtest.available_capital
        available_capital = backtest.available_capital
        for position in backtest.open_positions:
            realized_portfolio_value += position.size
            portfolio_value += position.size
            portfolio_value += position.unrealized_pnl
        self.PORTFOLIO.append({
            'close_time': backtest.current_ts,
            'available_capital': available_capital,
            'portfolio_value': portfolio_value,
            'realized_portfolio_value': realized_portfolio_value
        })
        return

    def save_closed_position(self, position):
        """Save a position that has been closed."""
        self.CLOSED_POSITIONS.append({
            'pair': position.pair,
            'side': position.side,
            'fee_rate': position.fee_rate,
            'entry_time': position.entry_time,
            'exit_time': position.exit_time,
            'entry_price': position.entry_price,
            'exit_price': position.exit_price,
            'size': position.size,
            'pnl': position.pnl,
            'pnl%': position.pnl_pct,
            'fees': position.fees,
            'success': position.success,
            'status': position.status,
            'highest_price_seen': position.highest_price_seen,
            'lowest_price_seen': position.lowest_price_seen,
            'drawdown': position.drawdown,
            'max_drawdown': position.max_drawdown
        })
        return

    def to_dataframes(self):
        """Build a nice dataframe out of all the tracked statistics"""
        self.DF_PORTFOLIO = pd.DataFrame(self.PORTFOLIO)
        self.DF_CLOSED_POSITIONS = pd.DataFrame(self.CLOSED_POSITIONS)
        self.DF_PORTFOLIO.to_csv(self.paths['portfolio'], sep=',', index=False)
        self.DF_CLOSED_POSITIONS.to_csv(self.paths['closed_positions'], sep=',', index=False)
        self.dataframes_created = True
        return

    def plot_portfolio_evolution(self):
        """Plot portfolio realized and unrealized values, with available capital over time."""
        if not self.dataframes_created:
            raise Exception("A backtest must be lead before plotting any portfolio evolution.")
        start_date = self.DF_PORTFOLIO['close_time'].min()
        end_date = self.DF_PORTFOLIO['close_time'].max()
        fig = plt.figure(figsize=(15, 6))
        ax = fig.add_subplot(111)
        # ax.plot(self.DF_PORTFOLIO['close_time'], self.DF_PORTFOLIO['available_capital'], linewidth=0.8, color='deepskyblue', label='available capital')
        ax.plot(self.DF_PORTFOLIO['close_time'], self.DF_PORTFOLIO['realized_portfolio_value'], linewidth=0.8, color='darkblue', label='realized portfolio value')
        ax.plot(self.DF_PORTFOLIO['close_time'], self.DF_PORTFOLIO['portfolio_value'], linewidth=0.8, color='red', label='unrealized portfolio value')
        ax.set_xlabel('close time', fontsize=15)
        ax.set_ylabel('quote currency', fontsize=15)
        ax.set_title(f'Evolution of portfolio during backtest from {start_date} to {end_date}', fontsize=18)
        ax.legend()
        plt.savefig(self.paths['portfolio_plot'], dpi=400, format='png')
        plt.close('all')
        return

    def plot_cumulated_profits(self):
        """Plot the cumulated profits of each pair, and of the whole universe."""
        if not self.dataframes_created:
            raise Exception("A backtest must be lead before plotting any profits evolution.")
        if (self.DF_CLOSED_POSITIONS.shape[0] == 0):
            return
        start_date = self.DF_PORTFOLIO['close_time'].min()
        end_date = self.DF_PORTFOLIO['close_time'].max()
        fig = plt.figure(figsize=(15, 6))
        ax = fig.add_subplot(111)
        # pair,side,fee_rate,entry_time,exit_time,entry_price,exit_price,size,pnl,pnl%,fees,success,status,highest_price_seen,lowest_price_seen,drawdown,max_drawdown
        for pair in list(set(self.DF_CLOSED_POSITIONS['pair'].values.tolist())):
            pair_df = self.DF_CLOSED_POSITIONS[self.DF_CLOSED_POSITIONS['pair'] == pair].copy()
            ax.plot(pair_df['exit_time'], pair_df['pnl'].cumsum(), linewidth=0.7, label=pair)
        ax.plot(self.DF_CLOSED_POSITIONS['exit_time'], self.DF_CLOSED_POSITIONS['pnl'].cumsum(), linewidth=1.2, color='darkblue', label='cumulated profits (all pairs)')
        ax.plot(self.DF_CLOSED_POSITIONS['exit_time'], [0]*len(self.DF_CLOSED_POSITIONS['exit_time']), linewidth=0.8, color='black')
        ax.set_xlabel('close time', fontsize=15)
        ax.set_ylabel('quote currency', fontsize=15)
        ax.set_title(f'Cumulated profits during backtest from {start_date} to {end_date}', fontsize=18)
        ax.legend()
        plt.savefig(self.paths['positions_profits_plot'], dpi=400, format='png')
        plt.close('all')
        return
    
    def compute_backtest_statistics(self, timedelta, initial_capital, risk_factor):
        """Compute win rate, avg_gain, avg_loss, expected_return, sharpe ratio, sortino ratio, max drawdown."""
        if not self.dataframes_created:
            raise Exception("A backtest must be lead before computing backtest statistics.")
        if self.DF_CLOSED_POSITIONS.shape[0] == 0:
            opt_data = {
                "sharpe_ratio": 0.,
                "sortino_ratio": 0.,
                "gini_coefficient": 1.,
                # "risk_of_ruin": ror,
                "annualized_roi": 0.,
                "max_drawdown": 0.,
                "n_trades": 0.,
                "mc_winned_trades": 0.,
                "mc_lost_trades": 0.,
                "win_rate": 0.,
                "avg_gain": 0.,
                "avg_loss": 0.,
                "expected_return": 0.
            }
            return opt_data
        TRADES = self.DF_CLOSED_POSITIONS.copy()  
        LONG = self.DF_CLOSED_POSITIONS[self.DF_CLOSED_POSITIONS['side'] == 'long'].copy()
        SHORT = self.DF_CLOSED_POSITIONS[self.DF_CLOSED_POSITIONS['side'] == 'short'].copy()
        one_year_duration = pd.Timedelta('1Y')
        backtest_duration = pd.Timestamp(self.DF_PORTFOLIO['close_time'].iloc[-1]) - pd.Timestamp(self.DF_PORTFOLIO['close_time'].iloc[0])
        # Annualized Returns On Investement
        overall_roi = statistics.compute_annualized_roi(TRADES['pnl'], initial_capital, backtest_duration)
        long_roi = statistics.compute_annualized_roi(LONG['pnl'], initial_capital, backtest_duration)
        short_roi = statistics.compute_annualized_roi(SHORT['pnl'], initial_capital, backtest_duration)
        # Number of trades
        n_overall_trades = TRADES.shape[0]
        n_long_trades = LONG.shape[0]
        n_short_trades = SHORT.shape[0]
        # Gini Coefficient
        gini_coef = statistics.compute_gini_coefficient(self.DF_CLOSED_POSITIONS)
        # Annualized Sharpe Ratio
        sharpe_ratio = statistics.compute_sharpe_ratio(self.DF_PORTFOLIO['portfolio_value'], timedelta)
        # Annualized Sortino Ratio
        sortino_ratio = statistics.compute_sortino_ratio(self.DF_PORTFOLIO['portfolio_value'], timedelta)
        # Win Rates
        global_win_rate = statistics.compute_win_rate(TRADES)
        long_win_rate = statistics.compute_win_rate(LONG)
        short_win_rate = statistics.compute_win_rate(SHORT)
        # Average Gains
        global_avg_gain = statistics.compute_avg_gain(TRADES)
        long_avg_gain = statistics.compute_avg_gain(LONG)
        short_avg_gain = statistics.compute_avg_gain(SHORT)
        # Average Loss
        global_avg_loss = statistics.compute_avg_loss(TRADES)
        long_avg_loss = statistics.compute_avg_loss(LONG)
        short_avg_loss = statistics.compute_avg_loss(SHORT)
        # Expected Returns
        global_expected_roi = statistics.compute_expected_roi(global_win_rate, global_avg_gain, global_avg_loss)
        long_expected_roi = statistics.compute_expected_roi(long_win_rate, long_avg_gain, long_avg_loss)
        short_expected_roi = statistics.compute_expected_roi(short_win_rate, short_avg_gain, short_avg_loss)
        # Max drawdowns
        max_drawdown = statistics.compute_max_drawdown(self.DF_PORTFOLIO['portfolio_value'])
        # Risk of a 10% Ruin
        # ror = statistics.compute_ror(10, global_win_rate, risk_factor, global_avg_loss)
        # Max number of consecutive winned/lost trades
        mc_winned_trades = statistics.compute_max_consecutive_trades(TRADES, success=1)
        mc_lost_trades = statistics.compute_max_consecutive_trades(TRADES, success=0)
        
        opt_data = {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "gini_coefficient": gini_coef,
            # "risk_of_ruin": ror,
            "annualized_roi": overall_roi,
            "max_drawdown": max_drawdown,
            "n_trades": n_overall_trades,
            "mc_winned_trades": mc_winned_trades,
            "mc_lost_trades": mc_lost_trades,
            "win_rate": global_win_rate,
            "avg_gain": global_avg_gain,
            "avg_loss": global_avg_loss,
            "expected_return": global_expected_roi
        }
        f = open(self.paths['backtest_results'], 'w')
        lines = []
        lines.append("----- BACKTEST RESULTS -----\n")
        lines.append(f"Sharpe Ratio: {sharpe_ratio}\n")
        lines.append(f"Sortino Ratio: {sortino_ratio}\n")
        lines.append(f"Gini coefficient: {gini_coef}\n")
        # lines.append(f"Risk of a 10% Ruin: {ror}%\n")
        lines.append("\n")
        lines.append(f"Annualized ROI: {overall_roi}%\n")
        lines.append(f"Max Drawdown: {max_drawdown}%\n")
        lines.append(f"Number of trades: {n_overall_trades}\n")
        lines.append(f"Maximum number of consecutive winned trades: {mc_winned_trades}\n")
        lines.append(f"Maximum number of consecutive lost trades: {mc_lost_trades}\n")
        lines.append("\n")
        lines.append(f"Win rate: {global_win_rate}%\n")
        lines.append(f"Average gain: {global_avg_gain}%\n")
        lines.append(f"Average loss: {global_avg_loss}%\n")
        lines.append(f"Expected Return: {global_expected_roi}%\n")
        lines.append("\n")
        lines.append("LONG TRADES:\n")
        lines.append(f"\tAnnualized long ROI: {long_roi}%\n")
        lines.append(f"\tNumber of long trades: {n_long_trades}\n")
        lines.append(f"\tWin rate: {long_win_rate}%\n")
        lines.append(f"\tAverage gain: {long_avg_gain}%\n")
        lines.append(f"\tAverage loss: {long_avg_loss}%\n")
        lines.append(f"\tExpected Return: {long_expected_roi}%\n")
        lines.append("\n")
        lines.append("SHORT TRADES:\n")
        lines.append(f"\tAnnualized short ROI: {short_roi}%\n")
        lines.append(f"\tNumber of short trades: {n_short_trades}\n")
        lines.append(f"\tWin rate: {short_win_rate}%\n")
        lines.append(f"\tAverage gain: {short_avg_gain}%\n")
        lines.append(f"\tAverage loss: {short_avg_loss}%\n")
        lines.append(f"\tExpected Return: {short_expected_roi}%\n")
        lines.append("----------------------------")
        f.writelines(lines)
        f.close()
        return opt_data





class Position():
    def __init__(self, pair, side, size, fee_rate, ohlc):
        self.pair = pair
        self.side = side
        self.fee_rate = fee_rate
        self.entry_time = ohlc['close_time'].iloc[-1]
        self.entry_price = ohlc['close_price'].iloc[-1]
        self.size = (1 - fee_rate) * size
        self.fees = fee_rate * size
        self.pnl = -self.fees
        self.unrealized_pnl = -self.fees
        self.status = 'open'
        self.highest_price_seen = self.entry_price
        self.lowest_price_seen = self.entry_price
        self.drawdown = 0
        self.max_drawdown = 0
        return

    def update_stats(self, ohlc):
        if self.status != 'open':
            raise Exception("Cannot update stats of a closed position")
        price = ohlc['close_price'].iloc[-1]
        if price > self.highest_price_seen:
            self.highest_price_seen = price
        if price < self.lowest_price_seen:
            self.lowest_price_seen = price
        if self.side == 'long':
            self.unrealized_pnl = -self.fees + (1 - self.fee_rate) * self.size * (price - self.entry_price) / self.entry_price
            self.drawdown = self.highest_price_seen - price
        if self.side == 'short':
            self.unrealized_pnl = -self.fees + (1 - self.fee_rate) * self.size * (self.entry_price - price) / self.entry_price
            self.drawdown = price - self.lowest_price_seen
        self.max_drawdown = np.max((self.max_drawdown, self.drawdown))
        return

    def close(self, ohlc):
        self.update_stats(ohlc)
        self.exit_time = ohlc['close_time'].iloc[-1]
        self.exit_price = ohlc['close_price'].iloc[-1]
        if self.side == 'long':
            self.pnl += (1 - self.fee_rate) * self.size * (self.exit_price - self.entry_price) / self.entry_price
            self.fees += self.fee_rate * self.size * (self.exit_price - self.entry_price) / self.entry_price
        if self.side == 'short':
            self.pnl += (1 - self.fee_rate) * self.size * (self.entry_price - self.exit_price) / self.entry_price
            self.fees += self.fee_rate * self.size * (self.entry_price - self.exit_price) / self.entry_price
        self.pnl_pct = self.pnl / (self.size / (1 - self.fee_rate))
        self.success = 1 if self.pnl > 0 else 0
        self.status = 'closed'
        new_cash = self.size + self.pnl
        return new_cash
    



class Backtester():
    def __init__(self, strategy, start_ts, end_ts, settings):
        """
        settings = {
            'timeframe',
            'universe',
            'initial_portfolio_value',
            'fee_rate'
        }
        """
        self.universe = settings['universe']
        self.timeframe = settings['timeframe']
        self.timedelta = pd.Timedelta(self.timeframe)
        self.portfolio_value = settings['initial_portfolio_value']
        self.available_capital = settings['initial_portfolio_value']
        self.initial_capital = settings['initial_portfolio_value']
        self.fee_rate = settings['fee_rate']
        self.strategy = strategy
        self.tracker = PerformanceTracker()
        self.open_positions = []
        self.closed_positions = []
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.current_ts = start_ts
        return

    def read_csv(self, pair):
        root = Path(os.getcwd())
        data_path = root.parent / 'data' / pair
        path = data_path / f'{pair.upper()}{self.timeframe}.csv'
        df = pd.read_csv(path, sep=',')
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df = df[df['close_time'] <= self.current_ts]
        df.index = [i for i in range(df.shape[0])]
        df = df[[
            'open_time', 'close_time', 
            'open_price', 'high_price', 'low_price', 'close_price',
            'quote_volume'
        ]].iloc[-3 * self.strategy.required_data_length:].copy()
        df.index = [i for i in range(df.shape[0])]
        return df

    def load_continuation_data(self):
        """Gather data up to the current time."""
        data = dict()
        for pair in self.universe:
            data[pair] = self.read_csv(pair)
            data[pair] = self.strategy.compute_indicators(data[pair])
        return data

    def manage_open_position(self, position, pair_data):
        """Where to close any existing positions."""
        if self.strategy.close_position(pair_data, position):
            new_cash = position.close(pair_data)
            self.closed_positions.append(position)
            self.tracker.save_closed_position(position)
            self.available_capital += new_cash
        return position

    def track_portfolio_value(self):
        """Sum up available capital, plus all positions'unrealized pnl and size"""
        self.portfolio_value = self.available_capital
        for position in self.open_positions:
            self.portfolio_value += position.size
            self.portfolio_value += position.unrealized_pnl
        return

    def cyclic_process(self):
        """Process to be done during each iteration of the backtest"""
        data = self.load_continuation_data()
        # Manage open positions
        still_open_positions = []
        for position in self.open_positions:
            position.update_stats(data[position.pair])
            position = self.manage_open_position(position, data[position.pair])
            if position.status == 'open':
                still_open_positions.append(position)
        self.open_positions = still_open_positions
        # Open new positions
        for pair in self.universe:
            if self.strategy.open_long(data[pair]):
                size = self.strategy.position_size(data[pair], self.portfolio_value)
                if size <= self.available_capital:                    
                    self.open_positions.append(
                        Position(
                            pair=pair,
                            side='long',
                            size=size,
                            fee_rate=self.fee_rate,
                            ohlc=data[pair]
                        )
                    )
                    self.available_capital -= size
            
            if self.strategy.open_short(data[pair]):
                size = self.strategy.position_size(data[pair], self.portfolio_value)
                if size <= self.available_capital:                    
                    self.open_positions.append(
                        Position(
                            pair=pair,
                            side='short',
                            size=size,
                            fee_rate=self.fee_rate,
                            ohlc=data[pair]
                        )
                    )
                    self.available_capital -= size
        # Increment current time and update portfolio value
        self.tracker.track_portfolio_values(self)
        self.track_portfolio_value()
        self.current_ts += self.timedelta
        return
        
    def execute(self):
        """Execute backtest."""
        while self.current_ts <= self.end_ts:
            self.cyclic_process()
        self.tracker.to_dataframes()
        self.tracker.plot_portfolio_evolution()
        self.tracker.plot_cumulated_profits()
        opt_data = self.tracker.compute_backtest_statistics(self.timedelta, self.initial_capital, self.strategy.risk_factor*100)
        return opt_data



# At each iteration:
#     track available_capital
#     track portfolio_value
#     track realized_portfolio value
#     save recently closed positions

# At the end of the process:
#     compute stats (win rate, avg_gain, avg_loss, expected_return, sharpe ratio, sortino ratio, max drawdown)

if __name__ == "__main__":
    t0 = tm.time()

    settings = {
        'timeframe': '4h',
        'universe': ['BTCUSDT', 'ETHUSDT', 'LINKUSDT'],
        'initial_portfolio_value': 1000,
        'fee_rate': 0.0004
    }
    start_ts = pd.Timestamp("2020-01-01 00:00:00")
    end_ts = pd.Timestamp("2021-01-01 00:00:00")
    
    # strategy_settings = {
    #     'contract_type': 'both', 
    #     'risk_factor': 0.001,
    #     'fast_window': 40,
    #     'slow_window': 103,
    #     'std_window': 20,
    #     'breakout_window': 2,
    #     'exit_multiplier': 3
    # }
    strategy_settings = dict()
    strategy = strategies.TrendFollowingStrategy(settings=strategy_settings)
    backtester = Backtester(strategy, start_ts, end_ts, settings)
    opt_data = backtester.execute()

    t1 = tm.time()

    print("Temps de calcul: ", t1-t0)