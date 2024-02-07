class MarketSignal:
    def __init__(
            self,
            date,
            x_day_moving_avg,
            x_day_moving_avg_grad,
            price_snapshot,
            price_snapshot_grad,
            prev_signals,
            stck_indx,
    ):
        self.date = date
        self.stck_indx = stck_indx
        self.x_day_moving_avg = x_day_moving_avg
        self.x_day_moving_avg_grad = x_day_moving_avg_grad
        self.price_snapshot = price_snapshot
        self.price_snapshot_grad = price_snapshot_grad
        self.mvng_avg_dist = self.price_snapshot - self.x_day_moving_avg
        self.num_of_days_abv_moving_avg = 0
        self.num_of_days_blw_moving_avg = 0
        self.num_of_days_since_max_price = 0
        self.num_of_days_since_min_price = 0
        self.prev_signals: list[MarketSignal] = prev_signals
        if self.mvng_avg_dist > 0:
            self.num_of_days_abv_moving_avg = self.num_of_days_abv_moving_avg+1
        elif self.mvng_avg_dist < 0:
            self.num_of_days_blw_moving_avg = self.num_of_days_blw_moving_avg+1
        if self.price_snapshot_grad>0:
            self.num_of_days_since_min_price = self.num_of_days_since_min_price + 1
        elif self.price_snapshot_grad<0:
            self.num_of_days_since_max_price = self.num_of_days_since_max_price + 1
        self.find_num_of_days_rltv_moving_avg()
        self.find_num_of_days_rltv_lcl_maxima()

    def find_num_of_days_rltv_moving_avg(self):
        cur_signal = self
        # if self.stck_indx == 159:
        #     print("oye", self.date,  cur_signal.mvng_avg_dist)
        for prev_signal in self.prev_signals:
            if prev_signal.mvng_avg_dist > 0 and cur_signal.mvng_avg_dist>0:
                self.num_of_days_abv_moving_avg = self.num_of_days_abv_moving_avg+1
            elif prev_signal.mvng_avg_dist <= 0 and cur_signal.mvng_avg_dist>0:
                self.num_of_days_abv_moving_avg=1
            elif prev_signal.mvng_avg_dist < 0 and cur_signal.mvng_avg_dist<0:
                self.num_of_days_blw_moving_avg = self.num_of_days_blw_moving_avg+1
            elif prev_signal.mvng_avg_dist >= 0 and cur_signal.mvng_avg_dist<0:
                self.num_of_days_blw_moving_avg=1

    def find_num_of_days_rltv_lcl_maxima(self):
        cur_signal = self
        for prev_signal in self.prev_signals:
            if prev_signal.price_snapshot_grad > 0 and cur_signal.price_snapshot_grad>0:
                self.num_of_days_since_min_price = self.num_of_days_since_min_price+1
            elif prev_signal.price_snapshot_grad <= 0 and cur_signal.price_snapshot_grad>0:
                self.num_of_days_since_min_price=1
            elif prev_signal.price_snapshot_grad < 0 and cur_signal.price_snapshot_grad<0:
                self.num_of_days_since_max_price = self.num_of_days_since_max_price+1
            elif prev_signal.price_snapshot_grad >= 0 and cur_signal.price_snapshot_grad<0:
                self.num_of_days_since_max_price=1

    
