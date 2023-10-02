import json
from apps.environment_simulator.models import (
    SimulatedStock,
    SimulatedStockCovariance,
    SimulatedStockBuffer,
    SimulatedStockXCommodity,
)
from apps.environment_simulator.models.simulated_commodity import SimulatedCommodityBuffer
from apps.environment_simulator.sevice_layer.commodity_simulator import CommoditySimulator, CommodityVaryData
import numpy as np
from datetime import datetime
from django.db.models import Q
import random



class StockSimulator:
    def __init__(self, latest_time_step: datetime, next_time_step: datetime, is_coupled = False):
        self.eps = 0.3
        self.stocks = SimulatedStock.objects.all()
        self.stocks = list(self.stocks)
        self.no_of_stocks = len(list(self.stocks))
        self.stocks_covariances: list[SimulatedStockCovariance] = SimulatedStockCovariance.objects.all()
        self.stock_x_commodities: list[SimulatedStockXCommodity] = SimulatedStockXCommodity.objects.all()
        self.stocks = random.sample(self.stocks, self.no_of_stocks)
        self.is_coupled = is_coupled

        self.affected_stocks = {}
        self.covariances_data = {}
        self.latest_time_step = latest_time_step
        self.next_time_step = next_time_step

        for a_covar in self.stocks_covariances:
            stock_a = a_covar.stock_a
            stock_b = a_covar.stock_b
            cvr = a_covar.factor

            if stock_a.id not in self.covariances_data:
                self.covariances_data[stock_a.id] = {}
            
            stock_a_covars = self.covariances_data[stock_a.id]
            if stock_b.id not in stock_a_covars:
                stock_a_covars[stock_b.id] = cvr
            
            if stock_b.id not in self.covariances_data:
                self.covariances_data[stock_b.id] = {}

            stock_b_covars = self.covariances_data[stock_b.id]
            if stock_a.id not in stock_b_covars:
                stock_b_covars[stock_a.id] = cvr

        stock_ids = []
        for z_stocks in self.stocks:
            stock_ids.append(z_stocks.id)
        
        latest_captures = SimulatedStockBuffer.objects.filter(
            Q(stock_id__in = stock_ids)
            & Q (captured_at = latest_time_step)
        ).all()

        existing_next_captures = SimulatedStockBuffer.objects.filter(
            Q(stock_id__in = stock_ids)
            & Q (captured_at = next_time_step)
        ).all()

        self.next_buffer = {}
        for next_capt in existing_next_captures:
            self.next_buffer[next_capt.stock_id] = next_capt

        self.mem_buffer = {}
        for capt in latest_captures:
            self.mem_buffer[capt.stock_id] = capt

        self.varied_stck_x_cmmdts_dict = {}
        self.cmmdts_x_stcks_covars = {}
        # for xstcm_var in self.stock_x_commodities:
        #     if xstcm_var.stock.id not in self.stcks_x_cmmdts_covars:
        #         self.stcks_x_cmmdts_covars[xstcm_var.stock.id] = {}
        #     stck_x_cmmdts_covars = self.stcks_x_cmmdts_covars[xstcm_var.stock.id]
        #     if xstcm_var.commodity.id not in stck_x_cmmdts_covars:
        #         stck_x_cmmdts_covars[xstcm_var.commodity.id] = xstcm_var.factor

        for xcmst_var in self.stock_x_commodities:
            if xcmst_var.commodity.id not in self.cmmdts_x_stcks_covars:
                self.cmmdts_x_stcks_covars[xcmst_var.commodity.id] = {}
            cmmdt_x_stcks_covars = self.cmmdts_x_stcks_covars[xcmst_var.commodity.id]
            if xcmst_var.stock.id not in cmmdt_x_stcks_covars:
                cmmdt_x_stcks_covars[xcmst_var.stock.id] = xcmst_var.factor

    def vary_stock_prices(self):
        if self.is_coupled:
            cmmdty_sims = CommoditySimulator(self.latest_time_step, self.next_time_step)
            varied_cmmdts: dict[int, CommodityVaryData] = cmmdty_sims.vary_commodity_prices()
            for cmmdty_id, vary_data in varied_cmmdts.items():
                if cmmdty_id in self.cmmdts_x_stcks_covars:
                    vrd_stcks = self.cmmdts_x_stcks_covars[cmmdty_id]
                    # for each of the stock in the stocks of its x_commodity
                    for v_stck_id in vrd_stcks:
                        if v_stck_id not in self.varied_stck_x_cmmdts_dict:
                            self.varied_stck_x_cmmdts_dict[v_stck_id] = {}
                        stcks_vary_data = self.varied_stck_x_cmmdts_dict[v_stck_id]
                        if cmmdty_id not in stcks_vary_data:
                            stcks_vary_data[cmmdty_id] = vary_data.perc_change * vrd_stcks[v_stck_id]
                            # vary the set price with each of these commodities variations


        updated_stcks = []
        grad_stocks: list[SimulatedStock] = []
        for stck in self.stocks:
            gradient = stck.price_gradient
            sd = stck.price_sd
            min_price = stck.min_price
            max_price = stck.max_price
            avg_forward_steps = stck.avg_forward_steps
            avg_backward_steps = stck.avg_backward_steps
            steps_left = stck.price_steps_left
            
            curr_price = self.mem_buffer[stck.id].price_snapshot

            next_grad_price = curr_price + gradient
            next_grad_price = np.random.normal(next_grad_price, sd)

            if next_grad_price <= min_price:
                next_grad_price = min_price
            elif next_grad_price >= max_price:
                next_grad_price = max_price

            price_diff = next_grad_price - curr_price
            perc_diff = price_diff/curr_price
            # print(f"{stck.id}, {stck.name}", curr_price, gradient, next_grad_price, perc_diff)
            if stck.id not in self.affected_stocks:
                self.affected_stocks[stck.id] = {}
            
            covars_data = self.covariances_data[stck.id]
            for stck_id, xvar in covars_data.items():
                if stck_id != stck.id:
                    if stck_id not in self.affected_stocks:
                        self.affected_stocks[stck_id] = {}
                    self.affected_stocks[stck_id][stck.id] = xvar*perc_diff
            
            # print(json.dumps(self.affected_stocks, indent=3))
            # print("")

            steps_left = steps_left - 1
            new_grad = gradient
            if steps_left <= 0:
                neg_gradient = gradient * -1
                new_grad = np.random.choice([gradient, neg_gradient])
                if new_grad <=0:
                    steps_left = avg_backward_steps
                else:
                    steps_left = avg_forward_steps

            stck.price_steps_left = steps_left
            stck.price_gradient = new_grad
            updated_stcks.append(stck)
            grad_stocks.append({
                "obj": stck,
                "next_grad_price": next_grad_price,
                "original_price": curr_price
            })

        # print(json.dumps(self.affected_stocks, indent=3))
        # print(grad_stocks)
        next_snapshots: list[SimulatedStockBuffer] = []
        for a_stock_data in grad_stocks:
            a_stck: SimulatedStock = a_stock_data["obj"]
            next_price = a_stock_data["next_grad_price"]
            set_price = a_stock_data["original_price"]
            extra_effects = self.affected_stocks[a_stck.id]
            for an_id, price_effect in extra_effects.items():
                next_price = next_price + price_effect*set_price

            if self.is_coupled:
                if a_stck.id in self.varied_stck_x_cmmdts_dict:
                    cmmdties_vars = self.varied_stck_x_cmmdts_dict[a_stck.id]
                    for cmmdty_x_id in cmmdties_vars:
                        cmm_price_effect = cmmdties_vars[cmmdty_x_id]
                        next_price = next_price + cmm_price_effect*set_price


            next_snapshots.append(
                SimulatedStockBuffer(
                    stock = a_stck,
                    captured_at = self.next_time_step,
                    price_snapshot = next_price
                )
            )
        un_accounted_for_stcks = []
        for unaccntd_stck in updated_stcks:
            if unaccntd_stck.id not in self.next_buffer:
                un_accounted_for_stcks.append(unaccntd_stck)
        
        un_accounted_for_snpshts = []
        for snpsht in next_snapshots:
            if snpsht.stock_id not in self.next_buffer:
                un_accounted_for_snpshts.append(snpsht)

        SimulatedStock.objects.bulk_update(un_accounted_for_stcks, [
            'price_steps_left', 
            'price_gradient', 
        ])
        SimulatedStockBuffer.objects.bulk_create(un_accounted_for_snpshts)