import pandas as pd
import psycopg2


def load_from_pg(table, host, port, dbname, user, password):
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
    )
    sql = f"SELECT * FROM {table};"
    df = pd.read_sql(sql, conn)
    conn.close()
    return df
