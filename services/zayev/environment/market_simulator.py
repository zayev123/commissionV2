import copy
from datetime import datetime
import gymnasium as gym
from matplotlib.dates import relativedelta
from gymnasium import spaces
import numpy as np
from asgiref.sync import sync_to_async
import asyncio
import psycopg2
import pandas as pd
import pytz

class MarketSimulator(gym.Env):
    def __init__(self, env_config: dict=None, data_frames = None):
        self.env_config = env_config
        self.n_step_stocks =  self.env_config.get("n_step_stocks")
        self.n_step_cmmdties =  self.env_config.get("n_step_cmmdties")
        self.__initial_balance = 100000
        self.action_space = self.__get_actn_shape()
        (stock_observation_space, commodity_observation_space, wallet_observation_space) = self.__get_obs_shape() 
        self.observation_space = spaces.Tuple((stock_observation_space, commodity_observation_space, wallet_observation_space))
        self.data_frames = data_frames
        self.max_episode_steps = self.env_config.get("max_episode_steps")
        self._print_output = self.env_config.get("print_output")
        self.reset()
        
    
    def __get_actn_shape(self):
        no_of_stocks = 5
        wallet = 1
        no_of_actions = no_of_stocks + wallet

        # actns = [list(range(-1000,1001,1)) for i in range(no_of_stocks)]
        # num_possible_values = 2001  # 2001 values from -1000 to 1000 (inclusive)
        # actn_shape = gym.spaces.MultiDiscrete(np.array([[1001, 2],[1001, 2],[1001, 2],[1001, 2],[1001, 2]]))
        # amount_shape = gym.spaces.MultiDiscrete([1001,1001,1001,1001,1001])



        lower_bounds = [-1] * no_of_actions
        upper_bounds = [1] * no_of_actions

        actn_shape = spaces.Box(low=np.array(lower_bounds), high=np.array(upper_bounds))
                
        return actn_shape
    
    def __get_obs_shape(self):
        no_of_stocks = 5
        stock_observation_shape = (no_of_stocks, 9 + self.n_step_stocks)
        stock_lower_bounds = np.zeros(stock_observation_shape)
        stock_upper_bounds = np.full(stock_observation_shape, float('inf'))
        
        no_of_cmmdts = 6
        commodity_observation_shape = (no_of_cmmdts, 2 + self.n_step_cmmdties) 

        commodity_lower_bounds = np.zeros(commodity_observation_shape)
        commodity_upper_bounds = np.full(commodity_observation_shape, float('inf'))

        stock_observation_space = spaces.Box(low=stock_lower_bounds, high=stock_upper_bounds)

        commodity_observation_space = spaces.Box(low=commodity_lower_bounds, high=commodity_upper_bounds)

        wallet_observation_space = spaces.Box(low=0, high=float('inf'), shape=(1,))

                    
        return (stock_observation_space, commodity_observation_space, wallet_observation_space)
    
    
    def get_prev_stock_prices_data(self, target_date):
        tme = target_date
        target_dates = [0 for nstk in range(self.n_step_stocks+1)]
        num_stocks = self.observation_space[0].shape[0]
        for nstk in range(self.n_step_stocks+1):
            target_dates[nstk] = tme
            tme = tme - relativedelta(hours=2, minutes=30)

        stock_prev_prices = np.zeros((num_stocks, self.n_step_stocks+1))
        
        for dt in range(self.n_step_stocks+1):
            a_date = target_dates[dt]
            a_stck_condition = self.stcks_buffer_df['captured_at'] == a_date
            a_filtered_stck_df = self.stcks_buffer_df.loc[a_stck_condition]
            a_stcks_buffer_df = a_filtered_stck_df.to_dict(orient='records')
            for a_stck in a_stcks_buffer_df:
                indx = a_stck["index"] -1
                stock_prev_prices[indx][dt] = a_stck["price_snapshot"]

        price_perc_changes = np.zeros((num_stocks, self.n_step_stocks))
        copy_prev_prices = copy.deepcopy(stock_prev_prices)
        cpy_prev_prices = np.zeros((num_stocks, self.n_step_stocks))
        for a_indx in range(num_stocks):
            for price_change_index in range(self.n_step_stocks):
                last_price_index = price_change_index+1
                cpy_prev_prices[a_indx][price_change_index] = copy_prev_prices[a_indx][last_price_index]
                if copy_prev_prices[a_indx][last_price_index] != 0.0:
                    price_perc_changes[a_indx][price_change_index] = (copy_prev_prices[a_indx][price_change_index] - copy_prev_prices[a_indx][last_price_index])/copy_prev_prices[a_indx][last_price_index]
                else:
                    price_perc_changes[a_indx][price_change_index] = 0.0
            # print("stock_prev_prices", stock_prev_prices)
        return cpy_prev_prices
            
    def get_prev_cmmdty_prices_data(self, target_date):
        tme = target_date
        target_dates = [0 for nstk in range(self.n_step_cmmdties+1)]
        num_cmmdties = self.observation_space[1].shape[0]
        for ncmdt in range(self.n_step_cmmdties+1):
            target_dates[ncmdt] = tme
            tme = tme - relativedelta(hours=2, minutes=30)

        cmmdty_prev_prices = np.zeros((num_cmmdties, self.n_step_cmmdties+1))
        
        for dt in range(self.n_step_cmmdties+1):
            a_date = target_dates[dt]
            a_cmmdty_condition = self.cmmdties_buffer_df['captured_at'] == a_date
            a_filtered_cmmdty_df = self.cmmdties_buffer_df.loc[a_cmmdty_condition]
            a_cmmdties_buffer_df = a_filtered_cmmdty_df.to_dict(orient='records')
            for a_cmmdty in a_cmmdties_buffer_df:
                indx = a_cmmdty["index"] -1
                cmmdty_prev_prices[indx][dt] = a_cmmdty["price_snapshot"]

        price_perc_changes = np.zeros((num_cmmdties, self.n_step_cmmdties))
        copy_prev_prices = copy.deepcopy(cmmdty_prev_prices)
        cpy_prev_prices = np.zeros((num_cmmdties, self.n_step_cmmdties))
        for a_indx in range(num_cmmdties):
            for price_change_index in range(self.n_step_cmmdties):
                last_price_index = price_change_index+1
                cpy_prev_prices[a_indx][price_change_index] = copy_prev_prices[a_indx][last_price_index]
                if copy_prev_prices[a_indx][last_price_index] != 0.0:
                    price_perc_changes[a_indx][price_change_index] = (copy_prev_prices[a_indx][price_change_index] - copy_prev_prices[a_indx][last_price_index])/copy_prev_prices[a_indx][last_price_index]
                else:
                    price_perc_changes[a_indx][price_change_index] = 0.0

        return cpy_prev_prices


    
    def get_the_state(self, init = False):
        num_stocks = self.observation_space[0].shape[0]
        num_commodities = self.observation_space[1].shape[0]
        num_stock_attributes = self.observation_space[0].shape[1]
        num_cmmdty_attributes = self.observation_space[1].shape[1]
        if init:
            stock_state = np.zeros((num_stocks, num_stock_attributes))
            commodity_state = np.zeros((num_commodities, num_cmmdty_attributes))
            wallet_state = np.array([0])
        else:
            (stock_state, commodity_state, wallet_state) = self.state

        target_date = self.the_current_time_step
        stck_condition = self.stcks_buffer_df['captured_at'] == target_date
        filtered_stck_df = self.stcks_buffer_df.loc[stck_condition]
        stcks_buffer_df = filtered_stck_df.to_dict(orient='records')
        prev_stck_prices_data = self.get_prev_stock_prices_data(target_date=target_date)

        cmmdty_condition = self.cmmdties_buffer_df['captured_at'] == target_date
        filtered_cmmdty_df = self.cmmdties_buffer_df.loc[cmmdty_condition]
        cmmdties_buffer_df = filtered_cmmdty_df.to_dict(orient='records')
        prev_cmmdty_prices_data = self.get_prev_cmmdty_prices_data(target_date=target_date)

        for a_stck in stcks_buffer_df:
            indx = a_stck["index"] -1
            if init:
                stock_state[indx][7] = a_stck["price_snapshot"]
                available_per_stock = self.__initial_balance/5
                if a_stck["price_snapshot"] == 0:
                    shares = 0
                else:
                    shares = available_per_stock/a_stck["price_snapshot"]
                stock_state[indx][8] = shares
            else:
                # set new shares in the step function
                stock_state[indx][7] = stock_state[indx][1]
            # stock_state[indx][0] = a_stck["captured_at"]
            stock_state[indx][1] = a_stck["price_snapshot"]
            stock_state[indx][2] = a_stck["volume"]
            stock_state[indx][3] = a_stck["bid_vol"]
            stock_state[indx][4] = a_stck["bid_price"]
            stock_state[indx][5] = a_stck["offer_vol"]
            stock_state[indx][6] = a_stck["offer_price"]
            
            prev_prices = prev_stck_prices_data[indx]
            ij = 9
            # print(f"prev_prices{indx}", prev_prices)
            for prev_price in prev_prices:
                stock_state[indx][ij] = prev_price
                ij = ij + 1

            # print(f"stock_state[{indx}]", stock_state[indx])
        

        for a_cmmdty in cmmdties_buffer_df:
            indx = a_cmmdty["index"] - 1
            # commodity_state[indx][0] = a_cmmdty["captured_at"]
            commodity_state[indx][1] = a_cmmdty["price_snapshot"]
            prev_prices = prev_cmmdty_prices_data[indx]
            ij = 2
            for prev_price in prev_prices:
                commodity_state[indx][ij] = prev_price
                ij = ij + 1
                    
        return (stock_state, commodity_state, wallet_state)

    
    def reset(self):
        the_current_time_step = self.env_config.get("the_current_time_step")
        __last_time_step = the_current_time_step + relativedelta(hours=505)
        self.the_current_time_step = pytz.utc.localize(datetime.strptime(str(the_current_time_step), '%Y-%m-%d %H:%M:%S'))
        self.__last_time_step = pytz.utc.localize(datetime.strptime(str(__last_time_step), '%Y-%m-%d %H:%M:%S'))
        str_time_step = str(self.the_current_time_step)
        self.__step_no = 0
        if self.data_frames!= None:
            self.stck_df = self.data_frames[0]
            self.cmmdties_df = self.data_frames[1]
            self.stcks_buffer_df = self.data_frames[2]
            self.cmmdties_buffer_df = self.data_frames[3]
        else:
            str_last_time_step = str(self.__last_time_step)
            db_params = self.env_config.get("db_params")
            self.db_conn = psycopg2.connect(**db_params)
            self.cursor = self.db_conn.cursor()

            stcks_query = """
                SELECT *
                FROM simulated_stocks
            """
            self.cursor.execute(stcks_query)
            stck_data = self.cursor.fetchall()
            column_names = [desc[0] for desc in self.cursor.description]
            self.stck_df = pd.DataFrame(stck_data, columns=column_names) 
            # self.stck_df = self.stck_df.to_dict(orient='records')

            stcks_buffer_query = f"""
                SELECT simulated_stocks_buffers.*, simulated_stocks.index
                FROM simulated_stocks_buffers 
                JOIN simulated_stocks on simulated_stocks.id = simulated_stocks_buffers.stock_id
                WHERE 
                captured_at >= '{str_time_step}' AND captured_at <= '{str_last_time_step}'
            """
            self.cursor.execute(stcks_buffer_query)
            stcks_buffer_data = self.cursor.fetchall()
            column_names = [desc[0] for desc in self.cursor.description]
            self.stcks_buffer_df = pd.DataFrame(stcks_buffer_data, columns=column_names)
            # self.stcks_buffer_df = self.stcks_buffer_df.to_dict(orient='records')

            cmmdties_buffer_query = f"""
                SELECT simulated_commodities_buffers.*, simulated_commodities.index
                FROM simulated_commodities_buffers 
                JOIN simulated_commodities on simulated_commodities.id = simulated_commodities_buffers.commodity_id
                WHERE 
                captured_at >= '{str_time_step}' AND captured_at <= '{str_last_time_step}'
            """
            self.cursor.execute(cmmdties_buffer_query)
            cmmdties_buffer_data = self.cursor.fetchall()
            column_names = [desc[0] for desc in self.cursor.description]
            self.cmmdties_buffer_df = pd.DataFrame(cmmdties_buffer_data, columns=column_names)
            # self.cmmdties_buffer_df = self.cmmdties_buffer_df.to_dict(orient='records')

            cmmdties_query = """
                SELECT *
                FROM simulated_commodities
            """
            self.cursor.execute(cmmdties_query)
            cmmdties_data = self.cursor.fetchall()
            column_names = [desc[0] for desc in self.cursor.description]
            self.cmmdties_df = pd.DataFrame(cmmdties_data, columns=column_names) 
            # self.cmmdties_df = self.cmmdties_df.to_dict(orient='records')

            self.cursor.close()
            self.db_conn.close()
        #return <obs>
        self.state = self.get_the_state(init=True)
        return self.state
                           
    def __get_action_change(self, action, index):
        actions = action
        change_in_shares = actions[index]
        # divisor = 2

        # factor = change_in_shares // divisor
        # remainder = change_in_shares % divisor

        # if remainder == 0:
        #     change_in_shares = -1*factor
        # else:
        #     change_in_shares = factor
        return change_in_shares

    
    def step(self, action):
        # if we took an action, we were in state 1
        copied_action = copy.deepcopy(action)
        self.the_current_time_step = self.the_current_time_step + relativedelta(hours=2, minutes=30)
        self.__step_no = self.__step_no + 1
        (stock_state, commodity_state, wallet_state) = self.get_the_state()
        if self._print_output:
            print("")
            print(f"stepping_a {action}")
            print(commodity_state)
            print(stock_state)
            print(wallet_state)


        no_of_actions = len(action)
        no_of_actions = no_of_actions - 1
        wallet_balance = wallet_state[0]
        old_portfolio_value = wallet_balance
        total_freed_capital = wallet_balance
        current_portfolio_value = wallet_balance
        new_portfolio_value = 0
        # total_value_bought = 0
        # total_value_sold = 0
        penalty = 0
        new_shares = {}
        flagged = False
        done = False
        trends = {}

        for index in range(no_of_actions):
            stck_price = stock_state[index][1]
            old_stock_price = stock_state[index][7]
            current_no_of_shares = stock_state[index][8]

            old_portfolio_value = old_portfolio_value + current_no_of_shares*old_stock_price
            current_portfolio_value = current_portfolio_value + current_no_of_shares*stck_price
            trends[index] = stock_state[index][12] - stock_state[index][9]

        for index in range(no_of_actions):
            bid_vol = stock_state[index][3]
            action_amount = copied_action[index]*current_portfolio_value

            bid_price = stock_state[index][4]
            offer_price = stock_state[index][6]
            if action_amount > 0:
                change_in_shares = action_amount/offer_price       
            else:
                change_in_shares = action_amount/bid_price

            action[index] = change_in_shares
            if self._print_output:
                print(f"change_in_shares for index: {index}", change_in_shares, current_no_of_shares, bid_vol)

        if self._print_output:
            print("nw_ports", current_portfolio_value)
            print("old_actions", copied_action)
            print("new_actions", action)

        for index in range(no_of_actions):
            stck_price = stock_state[index][1]
            stck_vol = stock_state[index][2]
            bid_vol = stock_state[index][3]
            bid_price = stock_state[index][4]
            offer_vol = stock_state[index][5]
            offer_price = stock_state[index][6]
            old_stock_price = stock_state[index][7]
            current_no_of_shares = stock_state[index][8]

            change_in_shares = action[index]
            
            if change_in_shares < 0:

                if bid_vol + change_in_shares < 0:
                    penalty = penalty + bid_vol + change_in_shares
                    flagged = True
                    if self._print_output:
                        print("break_2")
                    change_in_shares = -1*bid_vol

                if current_no_of_shares + change_in_shares < 0:
                    penalty = penalty + current_no_of_shares + change_in_shares
                    flagged = True
                    if self._print_output:
                        print("break_3")
                    change_in_shares = -1*current_no_of_shares + 1
                total_freed_capital = total_freed_capital + abs(change_in_shares)*bid_price
                new_shares[index] = change_in_shares + current_no_of_shares
            
            if change_in_shares == 0:
                new_shares[index] = current_no_of_shares

        for index in range(no_of_actions):
            stck_price = stock_state[index][1]
            stck_vol = stock_state[index][2]
            bid_vol = stock_state[index][3]
            bid_price = stock_state[index][4]
            offer_vol = stock_state[index][5]
            offer_price = stock_state[index][6]
            old_stock_price = stock_state[index][7]
            current_no_of_shares = stock_state[index][8]

            change_in_shares = self.__get_action_change(action, index)
                
            if change_in_shares > 0:
                if offer_vol - change_in_shares < 0:
                    penalty = penalty + offer_vol - change_in_shares
                    flagged = True
                    if self._print_output:
                        print("break_1")
                    change_in_shares = offer_vol
                    
                capital_locked = change_in_shares*offer_price
                if capital_locked > total_freed_capital:
                    penalty = penalty + total_freed_capital - capital_locked
                    capital_locked = total_freed_capital
                    if self._print_output:
                        print("crack_2")
                change_in_shares = capital_locked/offer_price
                total_freed_capital = total_freed_capital - capital_locked

                # total_value_bought = total_value_bought + capital_locked*offer_price
            
                new_shares[index] = change_in_shares + current_no_of_shares

        if current_portfolio_value == 0:
            done = True
            flagged = True
            penalty = penalty -10
            if self._print_output:
                print("break_4")
        
        # freed_balance = total_freed_capital - total_value_bought
        wallet_state[0] = total_freed_capital
        new_portfolio_value = total_freed_capital
        for index in range(no_of_actions):
            if new_shares[index] <= 0:
                new_shares[index] = 1
            stock_state[index][8] = new_shares[index]
            new_portfolio_value = new_portfolio_value + stock_state[index][8]*stock_state[index][1]

        # reward = current_portfolio_value - old_portfolio_value #+ penalty
        reward = self.get_trend_follow_reward(trends, action)

        self.state = (stock_state, commodity_state, wallet_state)
        info = {}

        if self.__step_no > self.max_episode_steps:
            done = True

        if self._print_output:
            print(f"stepping_b: {flagged} {old_portfolio_value}, {current_portfolio_value}, {new_portfolio_value} {penalty} {reward}")
            print(commodity_state)
            print(stock_state)
            print("")

        return self.state, reward, done, info
    
    def get_trend_follow_reward(self, trend, action):
        reward = 0
        for index in range(len(action)-1):
            sign = trend[index]*action[index] 
            if sign > 0:
                reward = reward +1
            else:
                reward = reward -1

        return reward



