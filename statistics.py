import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_gini_coefficient(df):
    """
    Calculate the Gini coefficient for measuring the consistency of returns.

    Arguments:
        df [pd.DataFrame]: must contain a 'pnl%' column and a 'exit_time' or 'close_time' column.
    """
    # The df must contain at least one line
    if df.shape[0] == 0:
        return 1.
    if not (('pnl%' in df.columns or 'portfolio_value' in df.columns) and ('exit_time' in df.columns or 'close_time' in df.columns)):
        raise Exception("[compute_gini_coefficient()]: the dataframe must contain either 'pnl%' and 'exit_time', or 'pnl%' and 'close_time' in its columns.")
    # Building the right proportion arrays for time and returns
    time_col = 'exit_time' if 'exit_time' in df.columns else 'close_time'
    profit_col = 'pnl%'
    # In case all trades were closed on the same datetime
    a = pd.Timestamp(df[time_col].min())
    b = pd.Timestamp(df[time_col].max())
    if a == b:
        return 1.
    c_times = np.array([(pd.Timestamp(df[time_col].iloc[i]) - a) / (b-a) for i in range(df.shape[0])])
    c_returns = (df[profit_col] - df[profit_col].min()).values
    c_returns = np.sort(c_returns)
    c_returns = np.cumsum(c_returns) / np.sum(c_returns)
    # Calculate the Gini coefficient
    G = 1
    for i in range(0, c_times.shape[0]-1):
        G -= (c_times[i+1] - c_times[i]) * (c_returns[i+1] + c_returns[i])
    G = round(G, 3)
    # Plot the cumulative distribution of relative profits
    fig = plt.figure(figsize=(18, 10))
    plt.plot(c_times, c_returns, linewidth=0.8, color='red')
    plt.plot(c_times, c_times, linewidth=0.8, color='black')
    plt.fill_between(x=c_times, y1=c_returns, y2=c_times, color='deepskyblue')
    plt.xlabel("backtest duration")
    plt.ylabel("Cumulated relative profits (ascending)")
    plt.title("Distribution of the profits through the backtest")
    plt.savefig('backtest_results/gini_coefficient.png', dpi=400, format='png')
    plt.close('all')
    return G


def compute_sharpe_ratio(portfolio_evolution: pd.Series, timedelta: pd.Timedelta):
    """
    Calculate the annualized Sharpe Ratio for measuring the returns relatively to the risk taken.
    
    Arguments:
        portfolio_evolution [pd.Series]: evolution of a portfolio value
        timedelta [pd.Timedelta]: timestep of the portfolio evolution
    """
    one_year_duration = pd.Timedelta('1Y')
    mean_return = portfolio_evolution.pct_change().mean()
    std_return = portfolio_evolution.pct_change().std()
    if std_return == 0:
        if mean_return == 0:
            return 0
        else:
            return 1
    sharpe_ratio = mean_return / std_return
    sharpe_ratio *= np.sqrt(one_year_duration / timedelta)
    sharpe_ratio = round(sharpe_ratio, 2)
    return sharpe_ratio
    

def compute_sortino_ratio(portfolio_evolution: pd.Series, timedelta: pd.Timedelta):
    """
    Calculate the annualized Sortino Ratio for measuring the return relatively to the negative risk taken.

    Arguments:
        portfolio_evolution [pd.Series]: evolution of a portfolio value
        timedelta [pd.Timedelta]: timestep of the portfolio evolution
    """
    one_year_duration = pd.Timedelta('1Y')
    mean_return = portfolio_evolution.pct_change().mean()
    negative_returns = portfolio_evolution.pct_change()
    negative_returns = negative_returns[negative_returns < 0].copy()
    std_negtive_returns = negative_returns.std()
    if std_negtive_returns == 0:
        if mean_return == 0:
            return 0
        else:
            return 1
    sortino_ratio = mean_return / std_negtive_returns
    sortino_ratio *= np.sqrt(one_year_duration / timedelta)
    sortino_ratio = round(sortino_ratio, 2)
    return sortino_ratio


def compute_ror(risked_loss, win_rate, risk_factor, avg_loss):
    """
    Calculate the Risk of Ruin of the strategy.

    Arguments:
        risked_loss [float]: portion of capital we want to measure the risk to loose (in %).
        win_rate [float]: proportion of winned trades (in %).
        risk_factor [float]: portion of capital we want to risk on each trade (in %).
        avg_loss [float]: average relative loss of lost trades (in %).

    """
    if (win_rate == 0):
        return 100
    if (avg_loss == 0):
        return 0
    u = (risked_loss/100) / (-avg_loss/100 * risk_factor/100)
    ror = ((1-win_rate/100)/(win_rate/100))**u
    ror = round(100 * ror, 2)
    return ror


def compute_win_rate(trades: pd.DataFrame):
    """
    Calculate the win rate from a table of trades.
    
    Arguments:
        trades [pd.DataFrame]: must contain a 'success' column.
    """
    if trades.shape[0] == 0:
        return 0
    if not 'success' in trades.columns:
        raise Exception("[compute_win_rate()]: no 'success' column in the given table.")
    wr = round(100 * trades['success'].sum() / trades.shape[0], 2)
    return wr


def compute_annualized_roi(pnls: pd.Series, initial_capital: float, backtest_duration: pd.Timedelta):
    """
    Calculate the annualized return of a strategy.
    
    Arguments:
        pnls [pd.Series]: array of the PNL over time.
        initial_capital [float]: initial amount of available capital.
        backtest_duration [pd.Timedelta]: duration of the backtest.
    """
    if initial_capital == 0:
        raise Exception("[compute_annualized_roi()]: initial_capital cannot be 0.")
    if backtest_duration == 0*pd.Timedelta('1M'):
        raise Exception("[compute_annualized_roi()]: backtest_duration cannot be 0.")
    one_year_duration = pd.Timedelta('1Y')
    roi = pnls.sum() / initial_capital
    roi = round(100 * ((1+roi)**(one_year_duration/backtest_duration) - 1), 2)
    return roi
    

def compute_avg_gain(trades: pd.DataFrame):
    """
    Calculate the avergae gain of some given trades, that is the average 
    relative PNL of the successful trades.
    
    Arguments:
        trades [pd.DataFrame]: table of trades.
    """
    if (not 'success' in trades.columns) or (not 'pnl%' in trades.columns):
        raise Exception("[compute_avg_gain()]: missing either 'success' or 'pnl%' column in the given trades dataframe.")
    if len(trades[trades['success'] == 1]) == 0:
        return 0.
    avg_gain = round(100 * trades[trades['success'] == 1]['pnl%'].mean(), 2)
    return avg_gain


def compute_avg_loss(trades: pd.DataFrame):
    """
    Calculate the avergae loss of some given trades, that is the average 
    relative PNL of the unsuccessful trades.
    
    Arguments:
        trades [pd.DataFrame]: table of trades.
    """
    if (not 'success' in trades.columns) or (not 'pnl%' in trades.columns):
        raise Exception("[compute_avg_gain()]: missing either 'success' or 'pnl%' column in the given trades dataframe.")
    if len(trades[trades['success'] == 0]) == 0:
        return 0.
    avg_loss = round(100 * trades[trades['success'] == 0]['pnl%'].mean(), 2)
    return avg_loss


def compute_expected_roi(win_rate: float, avg_gain: float, avg_loss:float):
    """
    Calculate the expected ROI of a given strategy.
    In other words, the relative average pnl per trade.
    
    Arguments:
        win_rate [float]: proportion of winned trades (in %)
        avg_gain [float]: average relative PNL of the winned trades (in %).
        avg_loss [float]: average relative PNL of the lost trades (in %).
    """
    expected_roi = (win_rate/100) * (avg_gain/100) + (1 - win_rate/100) * (avg_loss/100)
    expected_roi = round(100 * expected_roi, 2)
    return expected_roi


def compute_max_drawdown(portfolio_evolution: pd.Series):
    """
    Compute the maximum drawdown met in the given portfolio evolution.
    
    Arguments:
        portfolio_evolution [pd.Series]: evolution the portfolio value.
    """
    max_achieved_value = [np.max(portfolio_evolution.iloc[:i]) for i in range(1, len(portfolio_evolution))]
    max_achieved_value = np.array(max_achieved_value)
    max_drawdown = np.max(-(portfolio_evolution.iloc[1:].values / max_achieved_value - 1))
    max_drawdown = round(100 * max_drawdown, 2)
    return max_drawdown


def compute_max_consecutive_trades(trades, success):
    """
    Calculate the maximum number of consecutive winned or loss trades that happened during the backtest.

    Arguments:
        trades [pd.DataFrame]: table of closed trades.
        success [bool]: True for successful trades, False for unsuccessful ones.
    """
    max_trades = 1
    n_consecutive_trades = 0
    for i in range(trades.shape[0]):
        trade = trades.iloc[i]
        if (n_consecutive_trades == 0) and (trade['success'] == success):
            n_consecutive_trades += 1
        elif (n_consecutive_trades > 0) and (trade['success'] == success):
            n_consecutive_trades += 1
        max_trades = np.max([max_trades, n_consecutive_trades])
    return n_consecutive_trades
