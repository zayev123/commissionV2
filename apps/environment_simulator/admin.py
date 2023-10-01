from django.contrib import admin
from apps.environment_simulator.models import (
    SimulatedCommodity, 
    SimulatedCommodityCovariance,
    SimulatedCommodityBuffer,
    SimulatedSector, 
    SimulatedStock,
    SimulatedStockCovariance,
    SimulatedStockBuffer,
    SimulatedStockXCommodity
)

# Register your models here.
@admin.register(SimulatedCommodity)
class SimulatedCommodityAdmin(admin.ModelAdmin):
    list_display = [
        'index',
        'id', 
        'name', 
        'gradient', 
        'sd', 
        'min_price', 
        'max_price', 
        'avg_forward_steps', 
        'avg_backward_steps', 
        'steps_left',
    ]
    search_fields = ['index', 'name', 'avg_forward_steps']

@admin.register(SimulatedCommodityCovariance)
class SimulatedCommodityCovarianceAdmin(admin.ModelAdmin):
    list_display = [
        'commodity_a', 
        'commodity_b', 
        'factor', 
        'id', 
    ]
    search_fields = ['commodity_a__name', 'commodity_b__name']

@admin.register(SimulatedCommodityBuffer)
class SimulatedCommodityBufferAdmin(admin.ModelAdmin):
    list_display = [
        'captured_at',
        'commodity', 
        'price_snapshot', 
        'id', 
    ]
    search_fields = ['commodity__name', 'captured_at']

@admin.register(SimulatedSector)
class SimulatedSectorAdmin(admin.ModelAdmin):
    list_display = [
        'index', 
        'name', 
        'id', 
    ]
    search_fields = ['index', 'name']

@admin.register(SimulatedStock)
class SimulatedStockAdmin(admin.ModelAdmin):
    list_display = [
        'index', 
        'id', 
        'name', 
        'price_gradient',
        'price_sd',
        'min_price',
        'max_price',
        'volume_sd',
        'volume_x_price_factor',
        'price_x_volume_factor',
        'avg_forward_steps',
        'avg_backward_steps',
        'price_steps_left',

    ]
    search_fields = ['index', 'name']

@admin.register(SimulatedStockCovariance)
class SimulatedStockCovarianceAdmin(admin.ModelAdmin):
    list_display = [
        'stock_a', 
        'stock_b', 
        'factor', 
        'id', 
    ]
    search_fields = ['stock_a__name', 'stock_b__name']

@admin.register(SimulatedStockBuffer)
class SimulatedStockBufferAdmin(admin.ModelAdmin):
    list_display = [
        'captured_at',
        'stock', 
        'id', 
        'price_snapshot', 
        'ldcp',
        'open',
        'high',
        'low',
        'change',
        'volume'
    ]
    search_fields = ['stock__name', 'captured_at']

@admin.register(SimulatedStockXCommodity)
class SimulatedStockXCommodityAdmin(admin.ModelAdmin):
    list_display = [
        'stock', 
        'commodity', 
        'factor', 
        'id', 
    ]
    search_fields = ['stock__name', 'commodity__name']