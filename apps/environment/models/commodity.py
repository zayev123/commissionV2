from django.db import models


class Commodity(models.Model):
    symbol = models.CharField(max_length=300, blank=True, null=True)
    name = models.CharField(max_length=300)
    index = models.IntegerField(blank=True, null=True)
    units = models.CharField(max_length=30, blank=True, null=True)

    def __str__(self):
        return f"{str(self.id)}, " + str(self.index) + ', ' + str(self.name)
    
    class Meta:
        verbose_name_plural = "       commodities"
        ordering = ['index']
        db_table = 'commodities'
        app_label = 'environment'


class CommodityBuffer(models.Model):
    commodity = models.ForeignKey(Commodity, related_name='memory_snapshots', on_delete=models.CASCADE)
    captured_at = models.DateTimeField(blank=True, null=True)
    price_snapshot = models.FloatField(blank=True, null=True)


    def __str__(self):
        return f"{str(self.id)}, " + str(self.commodity.name) + ', ' + str(self.price_snapshot)
    
    class Meta:
        verbose_name_plural = "       commodities_buffers"
        ordering = ['-captured_at']
        db_table = 'commodities_buffers'
        app_label = 'environment'