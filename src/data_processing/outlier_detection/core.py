# src/data_processing/outlier_detection/core.py
"""
C2 强清洗：异常检测核心逻辑（不含统计与报告）

职责：
1. 对价格和成交量进行多维度异常检测；
2. 打标签：is_outlier_price / is_outlier_volume 等；
3. 不删除数据，也不覆盖原始价格 / 成交量。

统计与 PDF 报告请看同目录下的 report.py。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# -------------------------
# 常量与基础配置
# -------------------------

ROOT = Path(__file__).resolve().parents[3]  # 项目根目录

DATA_PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_INPUT_CSV = DATA_PROCESSED_DIR / "market_price_filled.csv"
DEFAULT_OUTPUT_CSV = DATA_PROCESSED_DIR / "market_price_cleaned.csv"

# 与 D 步保持一致的商品维度
GROUP_KEYS: List[str] = [
    "product_id",
    "variety",
    "spec",
    "grade",
    "shop_name",
    "classify_name",
    "color",
]

TS_COL = "ts"
PRICE_COL = "retail_price"
VOLUME_COL = "volume"


# -------------------------
# 内部辅助函数
# -------------------------


def _ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    """确保关键字段类型正确."""
    df = df.copy()
    df[TS_COL] = pd.to_datetime(df[TS_COL])

    # 价格和数量转为 float，方便后续计算
    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
    df[VOLUME_COL] = pd.to_numeric(df[VOLUME_COL], errors="coerce")

    return df


def _compute_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    按商品维度 + 时间序列计算基础特征：
    - 日涨跌幅
    - 7D/30D 滚动均值与标准差
    - Z 分数
    - 波动率
    - 跳变比例
    - 季节性 Z 分数
    """
    df = df.copy()
    df.sort_values(GROUP_KEYS + [TS_COL], inplace=True)

    g = df.groupby(GROUP_KEYS, dropna=False)

    # 日涨跌幅
    df["price_pct_change"] = g[PRICE_COL].pct_change()
    df["volume_pct_change"] = g[VOLUME_COL].pct_change()

    # 7D/30D 滚动均值与标准差（价格）
    df["price_mean_7d"] = g[PRICE_COL].transform(
        lambda s: s.rolling(window=7, min_periods=3).mean()
    )
    df["price_std_7d"] = g[PRICE_COL].transform(
        lambda s: s.rolling(window=7, min_periods=3).std(ddof=0)
    )
    df["price_mean_30d"] = g[PRICE_COL].transform(
        lambda s: s.rolling(window=30, min_periods=5).mean()
    )
    df["price_std_30d"] = g[PRICE_COL].transform(
        lambda s: s.rolling(window=30, min_periods=5).std(ddof=0)
    )

    # 成交量滚动统计
    df["volume_mean_7d"] = g[VOLUME_COL].transform(
        lambda s: s.rolling(window=7, min_periods=3).mean()
    )
    df["volume_std_7d"] = g[VOLUME_COL].transform(
        lambda s: s.rolling(window=7, min_periods=3).std(ddof=0)
    )

    # Z 分数
    df["z_price"] = (df[PRICE_COL] - df["price_mean_7d"]) / df["price_std_7d"]
    df["z_volume"] = (df[VOLUME_COL] - df["volume_mean_7d"]) / df["volume_std_7d"]

    # 涨跌幅波动率
    df["price_volatility_7d"] = g["price_pct_change"].transform(
        lambda s: s.rolling(window=7, min_periods=3).std(ddof=0)
    )
    df["price_volatility_30d"] = g["price_pct_change"].transform(
        lambda s: s.rolling(window=30, min_periods=5).std(ddof=0)
    )

    # 跳变比例（与前一日比较）
    prev_price = g[PRICE_COL].shift(1)
    df["jump_ratio"] = (df[PRICE_COL] - prev_price).abs() / prev_price.replace(0, np.nan)

    # 季节性（按 variety + 月份）
    if "variety" in df.columns:
        df["month"] = df[TS_COL].dt.month
        grp_season = df.groupby(["variety", "month"], dropna=False)[PRICE_COL]
        df["seasonal_mean_price"] = grp_season.transform("mean")
        df["seasonal_std_price"] = grp_season.transform("std").replace(0, np.nan)
        df["z_seasonal_price"] = (
            df[PRICE_COL] - df["seasonal_mean_price"]
        ) / df["seasonal_std_price"]
    else:
        df["z_seasonal_price"] = np.nan

    return df


def _rule_based_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于规则的异常标签：
    - Z 分数
    - 跳变比例
    - 涨跌幅
    - 波动率突增
    - 季节性偏离
    """
    df = df.copy()

    # 若存在节日标记，非节日才按严格规则判断
    if "holiday_flag" in df.columns:
        non_holiday = (df["holiday_flag"].fillna(0) == 0)
    else:
        non_holiday = pd.Series(True, index=df.index)

    # 价格异常规则
    cond_z_price = df["z_price"].abs() > 3
    cond_jump = df["jump_ratio"] > 2
    cond_pct_change = df["price_pct_change"].abs() > 1.5  # 涨跌幅 > 150%
    cond_vol_spike = (
        (df["price_volatility_7d"] > df["price_volatility_30d"] * 3)
        & df["price_volatility_30d"].notna()
        & (df["price_volatility_30d"] > 0)
    )
    cond_seasonal = df["z_seasonal_price"].abs() > 4

    price_outlier = (
        cond_z_price
        | cond_jump
        | cond_pct_change
        | cond_vol_spike
        | cond_seasonal
    ) & non_holiday

    # 成交量异常规则
    cond_z_vol = df["z_volume"].abs() > 3
    cond_vol_pct_change = df["volume_pct_change"].abs() > 2.0  # 成交量变化 > 200%

    volume_outlier = (cond_z_vol | cond_vol_pct_change) & non_holiday

    df["is_outlier_price"] = price_outlier.fillna(False).astype(bool)
    df["is_outlier_volume"] = volume_outlier.fillna(False).astype(bool)

    # 极端价格异常（可单独分析）
    df["is_extreme_price_outlier"] = (
        (df["z_price"].abs() > 5) | (df["jump_ratio"] > 3)
    ).fillna(False)

    return df


# -------------------------
# 对外主函数：异常检测
# -------------------------


def detect_outliers(
    df: pd.DataFrame,
    use_isolation_forest: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    对输入 DataFrame 进行异常检测，返回带标签的新 DataFrame。

    参数：
        df: 至少包含 ts / retail_price / volume 等字段
        use_isolation_forest: 是否额外启用 IsolationForest（可选）
    """
    df = _ensure_types(df)
    df = _compute_basic_features(df)
    df = _rule_based_outliers(df)

    if use_isolation_forest:
        try:
            from sklearn.ensemble import IsolationForest

            features = df[
                [
                    PRICE_COL,
                    VOLUME_COL,
                    "price_pct_change",
                    "volume_pct_change",
                    "price_volatility_7d",
                ]
            ].fillna(0)

            # 下采样以控制内存和速度
            max_samples = min(100_000, len(df))
            if len(df) > max_samples:
                rng = np.random.RandomState(random_state)
                sample_idx = rng.choice(df.index, size=max_samples, replace=False)
                fit_features = features.loc[sample_idx]
            else:
                fit_features = features

            iforest = IsolationForest(
                n_estimators=100,
                contamination=0.01,
                random_state=random_state,
                n_jobs=-1,
            )
            iforest.fit(fit_features)

            scores = iforest.decision_function(features)
            preds = iforest.predict(features)  # 1 正常，-1 异常

            df["iforest_score"] = scores
            df["is_iforest_outlier"] = preds == -1

            # 将 iforest 结果并入价格异常标签（保守 OR 合并）
            df["is_outlier_price"] = df["is_outlier_price"] | df["is_iforest_outlier"]

        except ImportError:
            # 未安装 sklearn 时自动跳过，不影响主流程
            df["iforest_score"] = np.nan
            df["is_iforest_outlier"] = False

    return df
