from django.db import models
from .simulated_stock import  SimulatedStock
from .simulated_commodity import SimulatedCommodity

class SimulatedStockXCommodity(models.Model):
    stock = models.ForeignKey(SimulatedStock, related_name='commodity_covariances', on_delete=models.CASCADE)
    commodity = models.ForeignKey(SimulatedCommodity, related_name='stock_covariances', on_delete=models.CASCADE)
    factor = models.FloatField(blank=True, null=True)

    def __str__(self):
        return f"{str(self.id)}, " + str(self.stock.name) + ', ' + str(self.commodity.name)

    class Meta:
        verbose_name_plural = "       simulated_stocks_x_commodities"
        db_table = 'simulated_stocks_x_commodities'
        app_label = 'environment_simulator'