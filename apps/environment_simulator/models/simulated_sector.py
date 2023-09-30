from django.db import models

class SimulatedSector(models.Model):
    name = models.CharField(max_length=300)
    index = models.IntegerField(blank=True, null=True)

    class Meta:
        verbose_name_plural = "       simulated_sectors"
        ordering = ['index']
        db_table = 'simulated_sectors'
        app_label = 'environment_simulator'