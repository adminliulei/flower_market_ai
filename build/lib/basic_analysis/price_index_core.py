# src/basic_analysis/price_index_core.py
"""
E 步：花价指数计算模块（basic_analysis）

目标：
1. 基于 C2 输出（market_price_cleaned.csv），计算每日的成交量加权平均价（VWAP）；
2. 支持三个层级的指数：
   - 全市场（ALL）
   - 按 classify_name（如：玫瑰、康乃馨等大类）
   - 按 variety（具体品种，如：红玫瑰，冰美人等）
3. 为后续特征工程和模型训练提供“市场趋势 / 行业基准”特征。

输出文件：
    data/intermediate/indices/flower_price_index.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# -------------------------
# 路径与常量
# -------------------------

ROOT = Path(__file__).resolve().parents[2]  # project root
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
INDICES_DIR = INTERMEDIATE_DIR / "indices"

DEFAULT_INPUT_CSV = PROCESSED_DIR / "market_price_cleaned.csv"
DEFAULT_OUTPUT_CSV = INDICES_DIR / "flower_price_index.csv"

TS_COL = "ts"
PRICE_COL = "retail_price"
VOLUME_COL = "volume"


@dataclass
class IndexSummary:
    """用于打印 / 报告的简单摘要"""

    total_rows: int
    scopes: List[str]
    date_range: str


# -------------------------
# 工具函数
# -------------------------

def _ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TS_COL] = pd.to_datetime(df[TS_COL])
    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
    df[VOLUME_COL] = pd.to_numeric(df[VOLUME_COL], errors="coerce")
    return df


def _calc_vwap(group: pd.DataFrame) -> pd.Series:
    """计算某个 group 的成交量加权平均价（VWAP）和总成交量。"""
    total_volume = group[VOLUME_COL].sum()
    if total_volume <= 0:
        return pd.Series({"price_index": np.nan, "total_volume": 0.0})
    total_value = (group[PRICE_COL] * group[VOLUME_COL]).sum()
    price_index = total_value / total_volume
    return pd.Series({"price_index": price_index, "total_volume": total_volume})


def _add_rolling_features(index_df: pd.DataFrame) -> pd.DataFrame:
    """
    按 scope_type + scope_value 维度，增加：
    index_ma7：7 日移动平均
    index_ma30：30 日移动平均
    index_return：日涨跌幅（不自动填充 NA）
    """
    df = index_df.copy()
    df.sort_values(["scope_type", "scope_value", TS_COL], inplace=True)

    g = df.groupby(["scope_type", "scope_value"], dropna=False)

    df["index_ma7"] = g["price_index"].transform(
        lambda s: s.rolling(window=7, min_periods=3).mean()
    )
    df["index_ma30"] = g["price_index"].transform(
        lambda s: s.rolling(window=30, min_periods=5).mean()
    )

    # ✅ 你要求的修改：不做 forward fill，保持 fill_method=None（更安全）
    df["index_return"] = g["price_index"].pct_change(fill_method=None)

    return df


# -------------------------
# 指数主计算逻辑
# -------------------------

def compute_price_index_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算长表形式的花价指数：
        ts, scope_type, scope_value,
        price_index, total_volume,
        index_ma7, index_ma30, index_return
    """
    df = _ensure_types(df)

    # --- 1）全市场指数（ALL）---
    overall = (
        df.groupby(TS_COL)[[PRICE_COL, VOLUME_COL]]
          .apply(_calc_vwap)
          .reset_index()
          .assign(scope_type="all", scope_value="ALL")
    )

    # --- 2）按 classify_name ---
    if "classify_name" in df.columns:
        by_classify = (
            df.groupby([TS_COL, "classify_name"])[[PRICE_COL, VOLUME_COL]]
              .apply(_calc_vwap)
              .reset_index()
              .rename(columns={"classify_name": "scope_value"})
              .assign(scope_type="classify")
        )
    else:
        by_classify = pd.DataFrame(columns=["ts", "scope_type", "scope_value", "price_index", "total_volume"])

    # --- 3）按 variety ---
    if "variety" in df.columns:
        by_variety = (
            df.groupby([TS_COL, "variety"])[[PRICE_COL, VOLUME_COL]]
              .apply(_calc_vwap)
              .reset_index()
              .rename(columns={"variety": "scope_value"})
              .assign(scope_type="variety")
        )
    else:
        by_variety = pd.DataFrame(columns=["ts", "scope_type", "scope_value", "price_index", "total_volume"])

    # --- 4）合并长表 ---
    index_df = pd.concat([overall, by_classify, by_variety], ignore_index=True)

    # 调整字段顺序
    index_df = index_df[
        [
            TS_COL,
            "scope_type",
            "scope_value",
            "price_index",
            "total_volume",
        ]
    ]

    # --- 5）增加 MA7 / MA30 / return ---
    index_df = _add_rolling_features(index_df)

    return index_df


# -------------------------
# 一站式 PIPELINE
# -------------------------

def run_price_index_pipeline(
    input_csv: Path = DEFAULT_INPUT_CSV,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
) -> IndexSummary:
    """完整执行 E 步：读入 → 计算指数 → 输出文件"""
    print(f"📥 E：加载 C2 输出数据：{input_csv}")
    df = pd.read_csv(input_csv)

    print("🧮 E：正在计算花价指数（VWAP） ...")
    index_df = compute_price_index_long(df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(output_csv, index=False)
    print(f"✅ E：花价指数已写入：{output_csv}")

    scopes = sorted(index_df["scope_type"].unique().tolist())
    date_range = f"{index_df[TS_COL].min()} ~ {index_df[TS_COL].max()}"

    return IndexSummary(
        total_rows=len(index_df),
        scopes=scopes,
        date_range=date_range,
    )
