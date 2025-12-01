"""
将原始行情表（例如 market_price、market_price_25_12、market_price_26_01 等）
增量同步到样本表 fm_market_price 的小工具。

表名全部从 .env 中读取：

- FM_TARGET_TABLE    -> 目标样本表（默认 fm_market_price）
- FM_SOURCE_TABLES   -> 源表列表，逗号分隔，如：
    FM_SOURCE_TABLES=market_price,market_price_25_12

使用方式（在项目根目录）：
    python -m src.utils.migrate_to_fm_market_price

也可以临时指定要同步的表（覆盖 .env）：
    python -m src.utils.migrate_to_fm_market_price --tables market_price_26_01
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import psycopg2
from psycopg2.extensions import connection as PGConnection

from config.settings import settings


# ----------------------------
# 可按需要微调的配置
# ----------------------------

# 自然唯一键字段（用于去重）
UNIQUE_KEY_COLUMNS: List[str] = [
    "ts", "product_id"
]

# 插入列（与 fm_market_price 中除 id 外的顺序一致）
INSERT_COLUMNS: List[str] = [
    "ts",
    "variety",
    "grade",
    "market_name",
    "wholesale_price",
    "retail_price",
    "volume",
    "classify_name",
    "spec",
    "stem_length_cm",
    "color",
    "product_id",
    "place",
    "shop_name",
    "image_url",
    "images",
    "unit",
    "ingest_at",
]


# ----------------------------
# 数据库连接
# ----------------------------

def get_conn() -> PGConnection:
    """
    从 config.settings 读取配置，建立 PostgreSQL 连接。
    需要在 .env 中配置：
        DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD
    """
    conn = psycopg2.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
    )
    conn.autocommit = True
    return conn


# ----------------------------
# 核心 SQL 生成与执行
# ----------------------------

def build_insert_sql(source_table: str, target_table: str) -> str:
    """
    生成从 source_table 增量插入 target_table 的 SQL。
    通过 LEFT JOIN + IS NULL 实现去重，不依赖唯一约束。
    """

    insert_cols_str = ", ".join(INSERT_COLUMNS)
    source_cols_str = ", ".join(f"s.{c}" for c in INSERT_COLUMNS)

    join_conditions = " AND ".join(
        f"COALESCE(s.{col}::text, '') = COALESCE(t.{col}::text, '')"
        for col in UNIQUE_KEY_COLUMNS
    )

    sql = f"""
    INSERT INTO {target_table} ({insert_cols_str})
    SELECT
        {source_cols_str}
    FROM {source_table} AS s
    LEFT JOIN {target_table} AS t
        ON {join_conditions}
    WHERE t.id IS NULL;
    """

    return sql


def migrate_one_table(conn: PGConnection, source_table: str, target_table: str) -> int:
    """
    将单个源表的数据增量写入目标样本表。
    返回插入的行数（可能为 -1 表示未知）。
    """
    sql = build_insert_sql(source_table, target_table)

    with conn.cursor() as cur:
        print(f"🚚 正在从 {source_table} 写入 {target_table} ...")
        cur.execute(sql)
        inserted = cur.rowcount if cur.rowcount is not None else -1

    print(f"✅ {source_table} -> {target_table} 完成，插入 {inserted} 行（去重后）")
    return inserted


# ----------------------------
# 源表列表获取
# ----------------------------

def resolve_source_tables(cli_tables: str | None) -> List[str]:
    """
    决定最终要同步的源表列表：

    1. 若命令行传入 --tables，则优先使用 CLI 参数（逗号分隔）
    2. 否则使用 settings.fm_source_tables（来自 .env -> FM_SOURCE_TABLES）
    """
    if cli_tables:
        tables = [t.strip() for t in cli_tables.split(",") if t.strip()]
        print(f"👉 使用命令行指定的源表列表：{tables}")
        return tables

    tables = settings.fm_source_tables
    print(f"👉 使用环境变量 FM_SOURCE_TABLES 指定的源表列表：{tables}")
    return tables


# ----------------------------
# CLI 入口
# ----------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="将原始行情表（market_price_XX）增量同步到样本表 fm_market_price"
    )
    parser.add_argument(
        "--tables",
        type=str,
        help=(
            "临时指定要同步的源表列表，逗号分隔，例如："
            "'market_price_25_12,market_price_26_01'。"
            "若不指定，则使用环境变量 FM_SOURCE_TABLES。"
        ),
    )
    args = parser.parse_args(argv)

    source_tables = resolve_source_tables(args.tables)
    target_table = settings.fm_target_table

    if not source_tables:
        print("⚠ 未发现需要同步的源表（FM_SOURCE_TABLES 为空？），直接退出。")
        return

    conn = get_conn()
    try:
        total_inserted = 0
        for tbl in source_tables:
            inserted = migrate_one_table(conn, tbl, target_table)
            if inserted and inserted > 0:
                total_inserted += inserted

        print(f"🎉 所有表同步完成，总插入行数：{total_inserted}")
    finally:
        conn.close()


if __name__ == "__main__":
    main(sys.argv[1:])
