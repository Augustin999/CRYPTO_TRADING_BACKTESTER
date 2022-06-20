# CRYPTO TRADING BACKTESTER

## Purpose of the project

This projct aims to develop an environment for automating strategy backtesting for cryptocurrency trading. Key features of the code are :
- Reusability: the general code must be capable of running backtests on different strategies, as long as strategies are based on OHLCV market data.
- Standardization: at the end of each backtest, the results and statistics must be available in the same form and contain the same stitistics. This means that all key figures will be annualized.
- Automation: A few lines of code will make backtesting possible.
- Multi-Market Backtesting: the simplest way to test a given strategy is on a single chosen market. With this project, a strategy can be run on multiple cryptocurrency markets, sharing the same investment capital. This is useful for diversification and choosing which cryptocurrencies to trade.
- Parallelized Optimization: a Genetic Algorithm based optimization class is useful for searching for the best settings for a given strategy. As the duration of a single backtest can be about 1 minute, the possibility of parallelizing backtests is the best option to spead up the optimization process.

## Steps

1. Create backtesting classes to lead backtests on multiple cryptocurrency markets at the same time, applying the same strategy for each one.
   1. The Strategy class is a template for defining strategies. It must contain position_size(), compute_indicators(), open_long(), open_short(), close_position() methods. All strategies that will be studied will be stored in the strategies file.
   2. The Position class is an implementation of a trade. A Position object is instanciated when a trade is strated. It can be updated along its lifetime, until it must be closed.
   3. The PerformanceTracker class is where the values of the portfolio are tracked along a backtest. It generates CSV files containing the closed positions and the evolution of the portfolio. The Performance Tracker also provide a method for computing multiple statistics and charts at the end of a backtest.
2. Create a Genetic Algorithm for optimizing strategies. This Genetic Algorithm must usable with different strategies. As each strategy can present a different number of indicators and parameters that can be optimized, the Genetic Algorithm must be a general frame. Only the fitness functions at most must be coded appropriately for each backtest.
3. Adapt the Genetic Algorithm for multiprocessing use. Parallelized computation will make running multiple backtests at the same time possible. 
4. Deal with overfitting issues. As overfitting is anticipated, tricks and technics must be searched for avoiding it. General preliminary ideas are walk-forward optimization, or using a different time interval for each generations. 
5. Add the capability of running an optimization on a Cloud Computing platform (GCP). 
