from django.db import models

from apps.scraper_pipes.models import Sector

class Stock(models.Model):
    name = models.CharField(max_length=300)
    index = models.IntegerField(blank=True, null=True)
    sector = models.ForeignKey(Sector, related_name='stocks', on_delete=models.CASCADE)

    def __str__(self):
        return f"{str(self.id)}, " + str(self.index) + ', ' + str(self.name)
    
    class Meta:
        verbose_name_plural = "       stocks"
        ordering = ['index']
        db_table = 'stocks'
        app_label = 'scraper_pipes'

class StockBuffer(models.Model):
    stock = models.ForeignKey(Stock, related_name='memory_snapshots', on_delete=models.CASCADE)
    captured_at = models.DateTimeField(blank=True, null=True)
    price_snapshot = models.FloatField(blank=True, null=True)
    change = models.FloatField(blank=True, null=True)
    volume = models.FloatField(blank=True, null=True)
    bid_vol = models.FloatField(blank=True, null=True)
    bid_price = models.FloatField(blank=True, null=True)
    offer_vol = models.FloatField(blank=True, null=True)
    offer_price = models.FloatField(blank=True, null=True)

    def __str__(self):
        return f"{str(self.id)}, " + str(self.stock.name) + ', ' + str(self.price_snapshot)


    class Meta:
        verbose_name_plural = "       stocks_buffers"
        ordering = ['-captured_at']
        db_table = 'stocks_buffers'
        app_label = 'scraper_pipes'