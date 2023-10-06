from django.db import models
from apps.environment_simulator.models import SimulatedStock
from apps.traders.models import Trader


class SimulatedOwnership(models.Model):
    stock = models.ForeignKey(SimulatedStock, related_name='owners', on_delete=models.CASCADE)
    trader = models.ForeignKey(Trader, related_name='owned_stocks', on_delete=models.CASCADE)
    shares = models.FloatField(blank=True, null=True)

    def __str__(self):
        return f"{str(self.id)}, " + str(self.stock.name) + ', ' + str(self.trader.name)
    
    class Meta:
        verbose_name_plural = "       simulated_ownerships"
        ordering = ['-shares']
        db_table = 'simulated_ownerships'
        app_label = 'traders'