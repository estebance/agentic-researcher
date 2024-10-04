import os
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

load_dotenv()

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}
CHECKPOINTER_PG_URL = os.environ.get("CHECKPOINTER_PG_URL")

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 1,
}

def retrieve_sync_connection_checkpointer():
    pool = ConnectionPool(
        # Example configuration
        conninfo=CHECKPOINTER_PG_URL,
        kwargs=connection_kwargs,
    )
    checkpointer = PostgresSaver(pool)
    return checkpointer
    # NOTE: you need to call .setup() the first time you're using your checkpointer
    # execute when required


def retrieve_async_connection_checkpointer():
    async_pool = AsyncConnectionPool(
        # Example configuration
        conninfo=CHECKPOINTER_PG_URL,
        max_size=20,
        kwargs=connection_kwargs,
    )
    async_checkpointer = AsyncPostgresSaver(async_pool)
    return async_checkpointer
    # NOTE: you need to call .setup() the first time you're using your checkpointer
    # await checkpointer.setup()

sync_checkpointer = retrieve_sync_connection_checkpointer()
# sync_checkpointer.setup()