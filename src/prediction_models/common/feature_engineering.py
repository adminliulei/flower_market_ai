# src/prediction_models/common/feature_engineering.py
"""
Feature Engineering for Flower Market AI

基于前置步骤：
- C1: preliminary_cleaning
- D : missing_value_filling
- C2: outlier_detection
- E : price_index (flower_price_index)

目标：
1. 从 C2 输出的 market_price_cleaned.csv 构造训练样本特征；
2. 将 E 步生成的花价指数（全市场 / 大类 / 品种）合并为特征；
3. 构造时间特征、滞后特征、滚动窗口特征、异常与波动特征；
4. 生成未来 1/2/3 天价格 & 成交量预测目标 (y)；
5. 输出到：
   - data/intermediate/features/time_series_features.csv
   - data/intermediate/features/category_features.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# -------------------------
# 路径 & 常量
# -------------------------

# ✅ 修正：项目根目录应该是 parents[3]（.../flower_market_ai）
ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
FEATURE_DIR = INTERMEDIATE_DIR / "features"
INDICES_DIR = INTERMEDIATE_DIR / "indices"

CLEANED_CSV = PROCESSED_DIR / "market_price_cleaned.csv"  # C2 输出
PRICE_INDEX_CSV = INDICES_DIR / "flower_price_index.csv"  # E 输出

TIME_SERIES_FEATURES_CSV = FEATURE_DIR / "time_series_features.csv"
CATEGORY_FEATURES_CSV = FEATURE_DIR / "category_features.csv"

TS_COL = "ts"
PRICE_COL = "retail_price"
VOLUME_COL = "volume"

GROUP_KEYS: List[str] = [
    "product_id",
    "variety",
    "spec",
    "grade",
    "shop_name",
    "classify_name",
    "color",
]


@dataclass
class FeatureSummary:
    """特征工程结果简要说明"""

    n_rows: int
    n_features: int
    date_range: str
    n_products: int
    targets: List[str]


# -------------------------
# 工具函数
# -------------------------

def _ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    """统一关键字段类型"""
    df = df.copy()
    df[TS_COL] = pd.to_datetime(df[TS_COL])
    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
    df[VOLUME_COL] = pd.to_numeric(df[VOLUME_COL], errors="coerce")
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """增加时间衍生特征"""
    df = df.copy()
    dt = df[TS_COL].dt

    df["day_of_week"] = dt.weekday  # 0-6, 周一=0
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["week_of_year"] = dt.isocalendar().week.astype(int)
    df["month"] = dt.month
    df["day_of_month"] = dt.day

    if "holiday_flag" in df.columns:
        df["holiday_flag"] = df["holiday_flag"].fillna(0).astype(int)
    else:
        df["holiday_flag"] = 0

    return df


def _add_lag_features(
    df: pd.DataFrame,
    lags_price: List[int] = [1, 2, 3, 7],
    lags_volume: List[int] = [1, 2, 3, 7],
) -> pd.DataFrame:
    """按商品维度增加价格 & 成交量滞后特征"""
    df = df.copy()
    df.sort_values(GROUP_KEYS + [TS_COL], inplace=True)

    g = df.groupby(GROUP_KEYS, dropna=False)

    for lag in lags_price:
        df[f"price_lag_{lag}"] = g[PRICE_COL].shift(lag)

    for lag in lags_volume:
        df[f"volume_lag_{lag}"] = g[VOLUME_COL].shift(lag)

    # 差分特征
    df["price_diff_1"] = df[PRICE_COL] - df["price_lag_1"]
    df["price_diff_7"] = df[PRICE_COL] - df["price_lag_7"]
    df["volume_diff_1"] = df[VOLUME_COL] - df["volume_lag_1"]

    return df


def _add_rolling_features(
    df: pd.DataFrame,
    price_windows: List[int] = [3, 7, 14, 30],
    volume_windows: List[int] = [7, 14],
) -> pd.DataFrame:
    """增加价格 / 成交量滚动窗口特征"""
    df = df.copy()
    df.sort_values(GROUP_KEYS + [TS_COL], inplace=True)

    g = df.groupby(GROUP_KEYS, dropna=False)

    for win in price_windows:
        df[f"price_ma_{win}"] = g[PRICE_COL].transform(
            lambda s, w=win: s.rolling(window=w, min_periods=max(2, w // 2)).mean()
        )
        df[f"price_std_{win}"] = g[PRICE_COL].transform(
            lambda s, w=win: s.rolling(window=w, min_periods=max(2, w // 2)).std(ddof=0)
        )
        df[f"price_cv_{win}"] = df[f"price_std_{win}"] / df[f"price_ma_{win}"]

    for win in volume_windows:
        df[f"volume_ma_{win}"] = g[VOLUME_COL].transform(
            lambda s, w=win: s.rolling(window=w, min_periods=max(2, w // 2)).mean()
        )
        df[f"volume_std_{win}"] = g[VOLUME_COL].transform(
            lambda s, w=win: s.rolling(window=w, min_periods=max(2, w // 2)).std(ddof=0)
        )

    if "price_ma_7" in df.columns and "price_ma_30" in df.columns:
        df["price_ma_ratio_7_30"] = df["price_ma_7"] / df["price_ma_30"]

    if "volume_ma_7" in df.columns and "volume_ma_14" in df.columns:
        df["volume_ma_ratio_7_14"] = df["volume_ma_7"] / df["volume_ma_14"]

    return df


def _load_price_index():
    """读取花价指数长表，并拆分为全市场 / 大类 / 品种三部分"""
    idx = pd.read_csv(PRICE_INDEX_CSV)
    idx["ts"] = pd.to_datetime(idx["ts"])

    # 全市场
    idx_all = (
        idx[idx["scope_type"] == "all"]
        .rename(
            columns={
                "price_index": "idx_all_price",
                "total_volume": "idx_all_volume",
                "index_ma7": "idx_all_ma7",
                "index_ma30": "idx_all_ma30",
                "index_return": "idx_all_return",
            }
        )
        .loc[
            :,
            [
                "ts",
                "idx_all_price",
                "idx_all_volume",
                "idx_all_ma7",
                "idx_all_ma30",
                "idx_all_return",
            ],
        ]
    )

    # 大类
    idx_cls = (
        idx[idx["scope_type"] == "classify"]
        .rename(
            columns={
                "scope_value": "classify_name",
                "price_index": "idx_cls_price",
                "total_volume": "idx_cls_volume",
                "index_ma7": "idx_cls_ma7",
                "index_ma30": "idx_cls_ma30",
                "index_return": "idx_cls_return",
            }
        )
        .loc[
            :,
            [
                "ts",
                "classify_name",
                "idx_cls_price",
                "idx_cls_volume",
                "idx_cls_ma7",
                "idx_cls_ma30",
                "idx_cls_return",
            ],
        ]
    )

    # 品种
    idx_var = (
        idx[idx["scope_type"] == "variety"]
        .rename(
            columns={
                "scope_value": "variety",
                "price_index": "idx_var_price",
                "total_volume": "idx_var_volume",
                "index_ma7": "idx_var_ma7",
                "index_ma30": "idx_var_ma30",
                "index_return": "idx_var_return",
            }
        )
        .loc[
            :,
            [
                "ts",
                "variety",
                "idx_var_price",
                "idx_var_volume",
                "idx_var_ma7",
                "idx_var_ma30",
                "idx_var_return",
            ],
        ]
    )

    return idx_all, idx_cls, idx_var


def _merge_price_index(df: pd.DataFrame) -> pd.DataFrame:
    """将花价指数（全市场 + 大类 + 品种）合并到明细数据上"""
    df = df.copy()
    idx_all, idx_cls, idx_var = _load_price_index()

    df = df.merge(idx_all, on="ts", how="left")

    if "classify_name" in df.columns:
        df = df.merge(idx_cls, on=["ts", "classify_name"], how="left")

    if "variety" in df.columns:
        df = df.merge(idx_var, on=["ts", "variety"], how="left")

    return df


def _add_targets(df: pd.DataFrame, horizons: List[int] = [1, 2, 3]) -> pd.DataFrame:
    """生成未来 1/2/3 天价格 & 成交量预测目标"""
    df = df.copy()
    df.sort_values(GROUP_KEYS + [TS_COL], inplace=True)

    g = df.groupby(GROUP_KEYS, dropna=False)

    for h in horizons:
        df[f"y_price_{h}d"] = g[PRICE_COL].shift(-h)
        df[f"y_volume_{h}d"] = g[VOLUME_COL].shift(-h)

    return df


def _build_category_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """构造类别字段的 ID 映射表"""
    cat_cols = ["variety", "classify_name", "grade", "color", "shop_name"]
    records = []

    for col in cat_cols:
        if col not in df.columns:
            continue
        uniq_vals = df[col].dropna().astype(str).unique()
        for idx, val in enumerate(sorted(uniq_vals)):
            records.append(
                {
                    "category_type": col,
                    "category_value": val,
                    "category_id": idx,
                }
            )

    mapping_df = pd.DataFrame(records)
    return mapping_df


# -------------------------
# 主流程：构建特征
# -------------------------

def build_features(
    cleaned_csv: Path = CLEANED_CSV,
    price_index_csv: Path = PRICE_INDEX_CSV,
) -> FeatureSummary:
    """核心入口：构造全部特征并输出 CSV"""
    if not cleaned_csv.exists():
        raise FileNotFoundError(f"Cleaned csv not found: {cleaned_csv}")
    if not price_index_csv.exists():
        raise FileNotFoundError(f"Price index csv not found: {price_index_csv}")

    print(f"📥 读取清洗后数据：{cleaned_csv}")
    df = pd.read_csv(cleaned_csv)
    df = _ensure_types(df)

    print("🧩 增加时间特征 ...")
    df = _add_time_features(df)

    print("🧩 合并花价指数特征（E 步） ...")
    df = _merge_price_index(df)

    print("🧩 构造价格/销量滞后特征 ...")
    df = _add_lag_features(df)

    print("🧩 构造滚动窗口特征 ...")
    df = _add_rolling_features(df)

    print("🎯 生成未来 1/2/3 天预测目标 (y) ...")
    df = _add_targets(df)

    target_cols = [c for c in df.columns if c.startswith("y_price_") or c.startswith("y_volume_")]
    df = df.dropna(subset=target_cols)

    print("🧩 生成类别映射表 ...")
    category_df = _build_category_mapping(df)

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"💾 写出特征数据：{TIME_SERIES_FEATURES_CSV}")
    df.to_csv(TIME_SERIES_FEATURES_CSV, index=False)

    print(f"💾 写出类别映射表：{CATEGORY_FEATURES_CSV}")
    category_df.to_csv(CATEGORY_FEATURES_CSV, index=False)

    date_range = f"{df[TS_COL].min()} ~ {df[TS_COL].max()}"
    summary = FeatureSummary(
        n_rows=len(df),
        n_features=df.shape[1],
        date_range=date_range,
        n_products=df["product_id"].nunique() if "product_id" in df.columns else 0,
        targets=target_cols,
    )
    return summary


# -------------------------
# 脚本入口
# -------------------------

def main():
    print("🌼 开始执行特征工程（Feature Engineering） ...")
    summary = build_features()

    print("\n📌 特征工程摘要：")
    print(f"- 样本行数：{summary.n_rows:,}")
    print(f"- 特征总数：{summary.n_features}")
    print(f"- 覆盖商品数：{summary.n_products}")
    print(f"- 日期范围：{summary.date_range}")
    print(f"- 目标列：{', '.join(summary.targets)}")
    print("\n✅ 特征工程完成，可用于 A/B 模型训练。")


if __name__ == "__main__":
    main()
