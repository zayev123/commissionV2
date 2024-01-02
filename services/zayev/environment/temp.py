class Temp:
    def step(self, action):
        # if the sign changes and a profit was made, then give it a reward, else dont do anything
        copied_action = copy.deepcopy(action)
        old_shares_data = copy.deepcopy(self.shares_data)
        
        self.__step_no = self.__step_no + 1
        (stck_state, cmmdty_state, volume_state) = copy.deepcopy(self.get_the_state())
        if self.__print_output:
            print("")
            print(f"stepping_a {action}")
            print(stck_state)
            print(cmmdty_state)
            print(volume_state)


        no_of_actions = len(action)
        no_of_actions = no_of_actions - 1
        if self.preparing:
            wallet_balance = self.__initial_balance
        else:
            wallet_balance = self.wallet_state
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

        reward = 0
        if self.change_in_shares:
            for indx in self.change_in_shares:
                reward = reward+ stck_state[indx-1][0]*self.change_in_shares[indx]
        
        for indx in range(no_of_actions):
            data_index = indx +1
            self.change_in_shares[data_index] = action[indx]

        # print(self.the_current_time_step, self.stock_data)
        for index in range(no_of_actions):
            data_index = index +1
            stck_price = self.stock_data[data_index]["price_snapshot"]
            chng = self.stock_data[data_index]["change"]
            old_stock_price = stck_price/(1+chng)
            current_no_of_shares = self.shares_data[data_index]

            old_portfolio_value = old_portfolio_value + current_no_of_shares*old_stock_price
            current_portfolio_value = current_portfolio_value + current_no_of_shares*stck_price

        for index in range(no_of_actions):
            data_index = index +1
            bid_vol = self.stock_data[data_index]["bid_vol"]
            action_amount = copied_action[index]*current_portfolio_value

            bid_price = self.stock_data[data_index]["bid_price"]
            offer_price = self.stock_data[data_index]["offer_price"]
            if action_amount > 0:
                change_in_shares = action_amount/offer_price       
            else:
                change_in_shares = action_amount/bid_price

            action[index] = change_in_shares
            if self.__print_output:
                print(f"change_in_shares for index: {index}", change_in_shares, current_no_of_shares, bid_vol)

        if self.__print_output:
            print("nw_ports", current_portfolio_value)
            print("old_actions", copied_action)
            print("new_actions", action)

        for index in range(no_of_actions):
            data_index = index +1
            stck_price = self.stock_data[data_index]["price_snapshot"]
            stck_vol = self.stock_data[data_index]["volume"]
            bid_vol = self.stock_data[data_index]["bid_vol"]
            bid_price = self.stock_data[data_index]["bid_price"]
            offer_vol = self.stock_data[data_index]["offer_vol"]
            offer_price = self.stock_data[data_index]["offer_price"]
            chng = self.stock_data[data_index]["change"]
            old_stock_price = stck_price/(1+chng)
            current_no_of_shares = self.shares_data[data_index]

            change_in_shares = action[index]
            
            if change_in_shares < 0:

                if bid_vol + change_in_shares < 0:
                    penalty = penalty + bid_vol + change_in_shares
                    flagged = True
                    if self.__print_output:
                        print("break_2")
                    change_in_shares = -1*bid_vol

                if current_no_of_shares + change_in_shares < 0:
                    penalty = penalty + current_no_of_shares + change_in_shares
                    flagged = True
                    if self.__print_output:
                        print("break_3")
                    change_in_shares = -1*current_no_of_shares + 1
                total_freed_capital = total_freed_capital + abs(change_in_shares)*bid_price
                new_shares[index] = change_in_shares + current_no_of_shares
            
            if change_in_shares == 0:
                new_shares[index] = current_no_of_shares

        for index in range(no_of_actions):
            data_index = index +1
            stck_price = self.stock_data[data_index]["price_snapshot"]
            stck_vol = self.stock_data[data_index]["volume"]
            bid_vol = self.stock_data[data_index]["bid_vol"]
            bid_price = self.stock_data[data_index]["bid_price"]
            offer_vol = self.stock_data[data_index]["offer_vol"]
            offer_price = self.stock_data[data_index]["offer_price"]
            chng = self.stock_data[data_index]["change"]
            old_stock_price = stck_price/(1+chng)
            current_no_of_shares = self.shares_data[data_index]

            change_in_shares = self.__get_action_change(action, index)
                
            if change_in_shares > 0:
                if offer_vol - change_in_shares < 0:
                    penalty = penalty + offer_vol - change_in_shares
                    flagged = True
                    if self.__print_output:
                        print("break_1")
                    change_in_shares = offer_vol
                    
                capital_locked = change_in_shares*offer_price
                if capital_locked > total_freed_capital:
                    penalty = penalty + total_freed_capital - capital_locked
                    capital_locked = total_freed_capital
                    if self.__print_output:
                        print("crack_2")
                change_in_shares = capital_locked/offer_price
                total_freed_capital = total_freed_capital - capital_locked

                # total_value_bought = total_value_bought + capital_locked*offer_price
            
                new_shares[index] = change_in_shares + current_no_of_shares

        if current_portfolio_value == 0:
            done = True
            flagged = True
            penalty = penalty -10
            if self.__print_output:
                print("break_4")
        
        # freed_balance = total_freed_capital - total_value_bought
        self.wallet_state = total_freed_capital
        new_portfolio_value = total_freed_capital
        for index in range(no_of_actions):
            data_index = index +1
            if new_shares[index] <= 0:
                new_shares[index] = 1
            self.shares_data[data_index] = new_shares[index]
            new_portfolio_value = new_portfolio_value + self.shares_data[data_index]*self.stock_data[data_index]["price_snapshot"]

        

        self.state = (stck_state, cmmdty_state, volume_state)
        info = {}

        if self.is_test:
            (is_valid_output, output) = self.test_output_validity(
                self.the_current_time_step, 
                self.shares_data, 
                old_shares_data,
                stck_state,
                self.stcks_buffer_df 
            )

            if not is_valid_output:
                print("test output:", output)

        if self.__step_no > self.max_episode_steps:
            done = True

        if self.__print_output:
            print(f"stepping_b: {flagged} {old_portfolio_value}, {current_portfolio_value}, {new_portfolio_value} {penalty} {reward}")
            print(self.stock_data)
            print(self.shares_data)
            print("")

        return self.state, reward, done, info