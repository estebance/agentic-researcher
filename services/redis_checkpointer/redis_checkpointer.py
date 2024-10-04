from .redis_saver import RedisSaver
import os
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv

load_dotenv()

REDIS_ENDPOINT = os.environ.get("REDIS_ENDPOINT")


def retrieve_sync_connection_checkpointer():
    checkpointer = RedisSaver.from_conn_info(host=REDIS_ENDPOINT, port=6379, db=0)
    return checkpointer