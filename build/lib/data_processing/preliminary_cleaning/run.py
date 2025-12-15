"""
C1 初步清洗运行入口脚本：

直接从 PostgreSQL 的样本表 fm_market_price 读取原始样本，
执行初步清洗（删除幽灵字段 + 单位统一 + 异常值处理），
并将结果导出为：

    data/processed/market_price_prelim_clean.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import psycopg2

from config.settings import settings
from .core import clean_preliminary


# 输出文件路径
ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "market_price_prelim_clean.csv"


def _load_from_pg() -> pd.DataFrame:
    """
    从 PostgreSQL 加载样本表 fm_market_price（或 .env 中配置的 fm_target_table）。
    """
    table_name = settings.fm_target_table  # 一般为 fm_market_price

    conn = psycopg2.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
    )
    try:
        sql = f"SELECT * FROM {table_name} ORDER BY ts, product_id;"
        df = pd.read_sql(sql, conn)
    finally:
        conn.close()

    return df


def main():
    print("📥 正在从 PostgreSQL 加载样本表 fm_market_price ...")
    df_raw = _load_from_pg()

    print(f"📊 原始样本数量：{len(df_raw):,} 行")
    print("🧹 开始执行 C1 初步清洗 ...")

    df_clean = clean_preliminary(df_raw)

    print(f"✅ C1 清洗完成：{len(df_clean):,} 行")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False)

    print(f"💾 清洗后数据已保存到：{OUTPUT_PATH}")



if __name__ == "__main__":
    main()
