import scrapy

class ScrStockBuffer(scrapy.Item):
    stock_name = scrapy.Field()
    captured_at = scrapy.Field()
    price_snapshot = scrapy.Field()
    change = scrapy.Field()
    volume = scrapy.Field()
    bid_vol = scrapy.Field()
    bid_price = scrapy.Field()
    offer_vol = scrapy.Field()
    offer_price = scrapy.Field()