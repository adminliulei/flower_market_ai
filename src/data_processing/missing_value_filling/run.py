"""
D 步运行入口脚本：

读取 C1 初步清洗后的数据：
    data/processed/market_price_prelim_clean.csv

执行整天缺失补全：
    fill_missing_days(df)

输出补全后的结果：
    data/processed/market_price_filled.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .core import fill_missing_days

# 项目根目录
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed"

INPUT_PATH = DATA_DIR / "market_price_prelim_clean.csv"
OUTPUT_PATH = DATA_DIR / "market_price_filled.csv"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"未找到 C1 输出文件：{INPUT_PATH}")

    print(f"📥 读取 C1 初步清洗后的数据：{INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    print("🧩 开始执行 D 步：整天缺失补全 ...")
    df_filled = fill_missing_days(df)

    print(f"✅ D 步完成：补全后总行数 {len(df_filled):,} 行")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_filled.to_csv(OUTPUT_PATH, index=False)
    print(f"💾 已保存到：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
