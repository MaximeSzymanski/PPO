import gymnasium as gym
import numpy as np
import yfinance as yf
import pandas as pd
import os
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from typing import List
from gymnasium import spaces
import matplotlib.pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter

import torch


# Define your trading strategy
def trading_strategy(data, portfolio_value):
    # Define your strategy logic here
    # Example: Buy if the current price is higher than the previous price, sell otherwise
    actions = []
    for i in range(1, len(data)):
        if data[i] > data[i - 1]:
            if portfolio_value >= 10 * data[i]:
                actions.append(("Buy", 10))
                portfolio_value -= 10 * data[i]
            else:
                actions.append(("Hold", 0))
        else:
            if len(actions) > 0 and actions[-1][0] == "Buy":
                actions.append(("Sell", 10))
                portfolio_value += 10 * data[i]
            else:
                actions.append(("Hold", 0))
    return actions




class StockEnv(gym.Env):
    """This class represents the environment of the stock market where the agent will learn to trade


        """

    def __init__(self, window_size=90, company_ticker=None):
        tickers = ['AIT', 'CSR', 'FTDR', 'JRVR', 'KAMN', 'MGPI', 'NKTR', 'OIS', 'PAHC', 'PRLB', 'TILE', 'TWNK', 'VIAV',
                   'WWW']

        number_train_tickers = 0.70 * len(tickers)
        number_test_tickers = len(tickers) - number_train_tickers
        print(
            f'Number of train tickers : {number_train_tickers} and number of test tickers : {number_test_tickers}')
        train_tickers = tickers[:int(number_train_tickers) + 1]
        # get the last two values of tje tickers array

        test_tickers = tickers[-int(number_test_tickers):]

        print(f'Train tickers : {train_tickers}')
        print(f'Test tickers : {test_tickers}')

        period = '5y'
        for ticker in train_tickers:
            df_temp = yf.Ticker(ticker).history(period=period)
            df_temp.drop(['Dividends', 'Stock Splits', 'Volume'],
                         axis=1, inplace=True)
            # create a folder for each stock
            df_temp.to_csv(f'./data_train/{ticker}.csv')
            print(f'{ticker} done')
        for ticker in test_tickers:
            df_temp = yf.Ticker(ticker).history(period=period)
            df_temp.drop(['Dividends', 'Stock Splits', 'Volume'],
                         axis=1, inplace=True)
            # create a folder for each stock
            df_temp.to_csv(f'./data_test/{ticker}.csv')
            print(f'{ticker} done')

        train_stocks = []
        test_stocks = []
        for ticker in train_tickers:
            # read the csv
            file_name = 'data_train/' + ticker + '.csv'
            df = pd.read_csv(file_name)
            train_stocks.append(Stock(ticker, df))
        for ticker in test_tickers:
            # read the csv
            file_name = 'data_test/' + ticker + '.csv'
            df = pd.read_csv(file_name)
            test_stocks.append(Stock(ticker, df))
        [stock.values.dropna() for stock in train_stocks]
        [stock.values.dropna() for stock in test_stocks]
        series_lengths_train = {len(series) for series in [
            stock.values for stock in train_stocks]}
        series_lengths_test = {len(series) for series in [
            stock.values for stock in test_stocks]}
        print(f'Train series lengths : {series_lengths_train}')
        print(f'Test series lengths : {series_lengths_test}')
        for stock in train_stocks:
            for index in range(1, len(stock.values)):
                stock.values.loc[index, 'DR'] = stock.values.loc[index, 'Close'] / stock.values.loc[index - 1, 'Close']
        for stock in test_stocks:
            for index in range(1, len(stock.values)):
                stock.values.loc[index, 'DR'] = stock.values.loc[index, 'Close'] / stock.values.loc[index - 1, 'Close']
        # normalize all datas between 0 and 100. Normalize independently all the stocks,
        scaler = MinMaxScaler(feature_range=(0, 10))

        # Train data normalization
        train_data_norm = pd.concat([stock.values.drop(['Date'], axis=1) for stock in train_stocks])
        train_data_norm = scaler.fit_transform(train_data_norm)
        train_data_norm_index = 0

        for stock in train_stocks:
            num_rows = len(stock.values)
            stock_values = train_data_norm[train_data_norm_index: train_data_norm_index + num_rows]
            df = pd.DataFrame(stock_values, columns=stock.values.columns[1:])
            stock.values.iloc[:, 1:] = df  # Update values for all columns except 'Date'
            train_data_norm_index += num_rows

        # Test data normalization
        test_data_norm = pd.concat([stock.values.drop(['Date'], axis=1) for stock in test_stocks])
        test_data_norm = scaler.transform(test_data_norm)
        test_data_norm_index = 0

        for stock in test_stocks:
            num_rows = len(stock.values)
            stock_values = test_data_norm[test_data_norm_index: test_data_norm_index + num_rows]
            df = pd.DataFrame(stock_values, columns=stock.values.columns[1:])
            stock.values.iloc[:, 1:] = df  # Update values for all columns except 'Date'
            test_data_norm_index += num_rows

        # Split the data into train and test
        self.test_stocks: List[Stock] = test_stocks
        self.train_stocks: List[Stock] = train_stocks
        # days is a list of all the days of the window size
        self.days = []
        self.company_ticker = company_ticker

        self.number_episode = 0

        self.window_size = window_size
        # There are 201 actions : 0 means no action, 1 to 100 is  buy (from 1 to 100 actions) and 101 to 201 means sell (from 101 to 201 actions)
        self.action_space = spaces.Discrete(3)
        self.hold_days = 0
        self.buy_days = 0
        self.sell_days = 0
        self.total_hold_days = 0
        self.total_days = 0
        self.interest = 0.0001

        """
        The observation space is a tuple of 6 elements :
            - The action space
            - The low value of the stock
            - The close value of the stock
            - The high value of the stock
            - The open value of the stock
            - The portfolio value
        """
        self.low_space = spaces.Box(low=0, high=1, shape=(1,))
        self.close_space = spaces.Box(low=0, high=1, shape=(1,))
        self.high_space = spaces.Box(low=0, high=1, shape=(1,))
        self.open_space = spaces.Box(low=0, high=1, shape=(1,))
        self.portfolio_space = spaces.Box(low=0, high=1000, shape=(1,))
        # The actions space is a box of 1 element. It represents the number of actions owned by the agent
        self.number_stocks_owned = spaces.Discrete(1000)
        self.current_stock_index = 0
        self.observation_space = spaces.Tuple((
            self.number_stocks_owned,
            self.low_space,
            self.close_space,
            self.high_space,
            self.open_space,
            self.portfolio_space
        ))
        self.reward_list = []
        self.iterator_stock = 0

    def add_day(self, daily_return, close, portfolio, stocks_owned):
        """
        Add a day to the list of days  and remove the first one if the list is too long.
        Arguments :
            - open : the open value of the stock
            - close : the close value of the stock
            - high : the high value of the stock
            - low : the low value of the stock
            - portfolio : the portfolio value
            - stocks_owned : the number of actions owned by the agent

        """
        # check if the list is too long
        if len(self.days) >= self.window_size:
            # Remove the first element
            self.days.pop(0)
        # append the new day at the end of the list
        # transform portofolio to a non tensor
        # check if the portfolio is a tensor
        if isinstance(portfolio, torch.Tensor):
            portfolio = portfolio.item()
        if isinstance(stocks_owned, torch.Tensor):
            actions = stocks_owned.item()

        self.days.append([daily_return, close / 10, portfolio / 10000, stocks_owned / 10])

    def _get_obs(self):
        # get the current state of the stock, and concatenate it with the portfolio and the actions

        return self.days

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the beginning of the stock values

        Arguments :
            - seed : the seed of the random generator
            - options : the options of the environment
        """

        self.reward = 0
        self.hold_days = 0
        # choose a random ticker from the train_stocks

        self.days = []
        # Define the walk-forward parameters
        # get a random stock from the train_stocks


        self.stock = self.train_stocks[self.current_stock_index]
        self.current_stock_index += 1
        if self.current_stock_index >= len(self.train_stocks):
            self.current_stock_index = 0

        # self.starting_point = random.randint(0, len(self.stock.values) - self.window_size - 1)
        self.starting_point = 0
        window_size_episode = 1300  # Window size for each optimization period
        self.window_size_episode = window_size_episode
        # if the starting point is too close to the end of the stock values, we adapt the window size episode
        if self.starting_point + window_size_episode > len(self.stock.values):
            window_size_episode = len(self.stock.values) - self.starting_point

        start_index = self.starting_point
        end_index = self.starting_point + window_size_episode
        data_period = self.stock.values.iloc[start_index:end_index + 1]['Close']
        # remove the index column of the series
        data_period = data_period.reset_index(drop=True)
        # Set the initial portfolio value
        initial_portfolio_value = 10000
        portfolio_value = initial_portfolio_value
        self.actions_to_do = trading_strategy(data_period, portfolio_value)

        # Print the results for the chosen period
        # print(f"Period: {self.starting_point} to {self.starting_point + window_size_episode - 1}")
        self.iterator_stock += 1
        if self.iterator_stock == 2:
            self.iterator_stock = 0
        self.portfolio_after = 0
        # self.stock = self.train_stocks[1]
        for i in range(self.window_size):
            self.add_day(0, 0, 0, 0)
        # choose a random starting point in the stock, but not too close to the end (window_size)
        # self.starting_point = random.randint(0, len(self.stock.values) - self.window_size - 1)
        self.diff_pourcentage = (
                (self.stock.values['Close'].iloc[-1] - self.stock.values['Close'].iloc[0]) /
                self.stock.values['Close'].iloc[0])
        self.min_close = self.stock.values['Close'].min()
        self.max_close = self.stock.values['Close'].max()
        """self.max_portfolio = (
            (self.max_close-self.min_close)/self.min_close) * 100 * 100 * self.min_close
        print(f'max portfolio : {self.max_portfolio}')"""
        # start at a random point in the stock

        # we assume that an episode is 512 step long
        # we will compute the reward with Walk-forward optimization example

        self.current_iteration = 0
        self.current_step = self.starting_point
        self.window_size = self.window_size
        self.portfolio = 10000
        self.actions = 0
        self._get_stocks_features()
        info = None
        if np.isnan(self.daily_return):
            self.daily_return = 0
        self.add_day(self.daily_return, self.close,
                     self.portfolio, self.actions)

        return self._get_obs(), info

    def _get_stocks_features(self):
        # geet the stocks features of the window, and the next day

        self.low = self.stock.values['Low'][self.current_step]
        self.close = self.stock.values['Close'][self.current_step]
        self.high = self.stock.values['High'][self.current_step]
        self.open = self.stock.values['Open'][self.current_step]
        self.daily_return = self.stock.values['DR'][self.current_step]

    def _evaluate_action(self, number_of_stocks):
        reward = 0
        is_finish = False

        # check if the action is valid
        if number_of_stocks == 0:
            self.total_hold_days += 1
            self.hold_days += 1
            # hold
            # check if the action of the next day is higher than the current day
            # add the interest to the portfolio

            # remove the interest from the portfolio

        elif number_of_stocks == 1:
            self.buy_days += 1
            self.hold_days = 0
            # buy
            if self.portfolio >= 10 * self.current_price:
                self.portfolio -= 10 * self.current_price
                self.actions += 10
            else:
                print('Not enough money')
                # self.reward -= number_of_stocks * 1000
                is_finish = True
                # not enough money
                pass
        else:
            self.hold_days = 0
            self.sell_days += 1
            number_of_stocks = 10
            # sell
            if self.actions >= number_of_stocks:
                self.portfolio += 10 * self.current_price
                self.actions -= 10

            else:
                print('not enough stocks')
                # self.reward -= number_of_stocks * 1000

                is_finish = True
                # not enough stocks
                pass

        return is_finish, reward

    def _compute_reward(self, action):

        # compute the reward of the action using the difference between the action and the Walk-forward optimization actoin
        if action == 0:
            if self.actions_to_do[self.current_iteration][0] == "Hold":
                self.reward = 1
            else:
                self.reward = 0
        elif action == 1:

            if self.actions_to_do[self.current_iteration][0] == "Buy":
                self.reward = 1
            else:
                self.reward = 0
        else:
            if self.actions_to_do[self.current_iteration][0] == "Sell":
                self.reward = 1
            else:
                self.reward = 0

    def _is_done(self):
        # print(self.current_step+1, len(self.stock.values))
        return self.current_step + 1 in [
            len(self.stock.values),
            self.starting_point + self.window_size_episode,
        ]

    def step(self, action):

        self.reward = 0
        # action is a tuple of (buy_sell_hold, number_of_stocks)
        # buy_sell_hold : 0 : hold, 1 : buy, 2 : sell
        # number_of_stocks : integer between 0 and 99
        self.current_price = self.stock.values['Close'][self.current_step]
        portfolio_before = self.portfolio + self.actions * self.current_price


        self.next_day_price = self.stock.values['Close'][self.current_step + 1]
        number_of_stocks = action
        is_finish, hold_reward = self._evaluate_action(number_of_stocks)

        # aplly interest penalty
        self.portfolio = self.portfolio * (1 - self.interest)

        # self._compute_reward(action)
        if is_finish:
            print("ERROR")
        info = {}
        self.current_step += 1
        self.current_iteration += 1
        done = self._is_done()
        if done:
            # sell the stocks
            self.portfolio += self.actions * self.next_day_price
            # self.writer.add_scalar("End portfolio value", self.portfolio, self.number_episode)

            self.actions = 0
            self.number_episode += 1

        self._get_stocks_features()
        # print('reward step : ', self.current_step, reward)
        self.add_day(self.daily_return, self.close,
                     self.portfolio, self.actions)
        # print('day :',self.days)

        info['portfolio'] = self.portfolio
        info['actions'] = self.actions * self.next_day_price

        if is_finish:
            # done = True
            # self.render(
            pass
        if done:
            # self.render()
            pass
        portfolio_after = self.portfolio + self.actions * self.next_day_price
        # done = done or is_finish
        # self.writer.add_scalar("hold per day", self.total_hold_days, self.total_days)
        # elf.writer.add_scalar("buy per day", self.buy_days, self.total_days)
        # self.writer.add_scalar("sell per day", self.sell_days, self.total_days)
        # print(self.portfolio_after)
        self.total_days += 1
        # self.writer.add_scalar("portfolio", portfolio_after, self.total_days)
        # self.writer.add_scalar("portfolio diff", portfolio_after - portfolio_before, self.total_days)
        if portfolio_after == 0:
            portfolio_after = 1
        self.portfolio_after = portfolio_after
        self.reward = portfolio_after - portfolio_before
        # put reward as a non tensor
        # put reward as a non tensor
        self.reward_list.append(self.reward)

        return self._get_obs(), self.reward, done, False, info

    def render(self, mode='human', close=False):
        plt.plot(self.portfolio)
        plt.show()
        pass

    def close(self):
        pass

@dataclass
class Stock:
    """This class represents a stock with its ticker and its values (Open, High, Low, Close)

    Attributes:
        ticker (str): the ticker of the stock
        values (pd.DataFrame): the values of the stock
        """
    ticker: str
    values: pd.DataFrame