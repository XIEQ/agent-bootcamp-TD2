from invoke import task
from src.eda_rajiv.utils.create_fintran_db import (
    db_get_connection,
    db_create_schema,
    db_load_data,
    db_run_query,
)


@task
def db_create_and_load_dataset(c, data_folder_path, db_path, table=None):
    conn = db_get_connection(db_path)
    db_create_schema(conn)
    tables = [table] if table else []
    db_load_data(conn, data_folder_path, tables=tables)


@task
def db_query(c, query, db_path, limit=5):
    conn = db_get_connection(db_path)
    df = db_run_query(conn, query)
    print(df)
