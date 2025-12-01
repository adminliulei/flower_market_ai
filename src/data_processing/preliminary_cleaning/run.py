import pandas as pd
from src.utils.db_utils import load_from_pg
from src.data_processing.preliminary_cleaning.core import clean_preliminary
from config.settings import settings
from pathlib import Path


OUTPUT_PATH = Path("data/processed/market_price_prelim_clean.csv")


def main():
    print("📥 正在从 PostgreSQL 加载样本表 fm_market_price ...")

    df = load_from_pg(
        table=settings.fm_target_table,  # 默认 fm_market_price
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
    )

    print(f"📊 原始样本数量：{len(df)} 行")
    print("🧹 开始执行 C1 初步清洗 ...")

    df_clean = clean_preliminary(df)

    print(f"✅ C1 清洗完成：{len(df_clean)} 行")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False)

    print(f"💾 清洗后数据已保存到：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
