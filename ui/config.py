import os

REDIS_HOST = os.environ.get('REDIS_HOST', 'testing-redis.xgwits.0001.usw2.cache.amazonaws.com')
REDIS_PORT = os.environ.get('REDIS_PORT', '6379')
REDIS_DB = os.environ.get('REDIS_DB', '7')
