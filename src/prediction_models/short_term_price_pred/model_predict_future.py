# src/prediction_models/short_term_price_pred/model_predict_future.py
"""
短期价格预测 - 真实未来推理脚本（不依赖 y_true）

用途：
    - 模拟真实线上预测：只用特征，不看任何未来真实价格
    - 对每个商品当前最新日期，预测未来 1/2/3 日价格

输入：
    data/intermediate/features/time_series_features.csv  （仅用特征列）

输出：
    data/output/price_forecast_future.csv

运行方式（项目根目录）：
    python -m src.prediction_models.short_term_price_pred.model_predict_future
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

from .model_train import load_feature_data, MODEL_ROOT, TS_COL

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "price_forecast_future.csv"


# 与特征工程里的商品维度一致
GROUP_KEYS: List[str] = [
    "product_id",
    "variety",
    "spec",
    "grade",
    "shop_name",
    "classify_name",
    "color",
]


def build_latest_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    从特征表中选出：每个商品维度的“最新日期”那一行特征。
    """
    df = df.copy()
    df.sort_values(TS_COL, inplace=True)

    group_cols = [c for c in GROUP_KEYS if c in df.columns]
    if not group_cols:
        # 没有这些字段，就按 ts 取最后一行
        latest_df = df.sort_values(TS_COL).tail(1)
    else:
        idx = (
            df.groupby(group_cols)[TS_COL]
            .idxmax()
            .dropna()
            .astype(int)
        )
        latest_df = df.loc[idx].copy()

    return latest_df


def prepare_feature_matrix(latest_df: pd.DataFrame) -> pd.DataFrame:
    """
    删除所有目标列，只保留特征列。
    对 object 列做 category.codes 编码。
    """
    df_feat = latest_df.copy()

    # 丢掉任何 y_* 目标列
    target_cols = [c for c in df_feat.columns if c.startswith("y_price_") or c.startswith("y_volume_")]
    df_feat = df_feat.drop(columns=target_cols, errors="ignore")

    # 特征列 = 除 ts 外的其他列
    feature_cols = [c for c in df_feat.columns if c != TS_COL]

    X = df_feat[feature_cols].copy()

    for col in feature_cols:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category").cat.codes.astype("int32")

    return X, feature_cols


def main():
    print("🌼 短期价格预测 - 真实未来推理开始 ...")

    # 1. 加载特征数据
    df = load_feature_data()

    # 2. 取每个商品最新一天的特征行
    latest_df = build_latest_feature_rows(df)
    print(f"🔎 最新特征行数（按商品维度去重）：{len(latest_df):,}")

    # 3. 准备特征矩阵
    X, feature_cols = prepare_feature_matrix(latest_df)

    # 4. 对 1/2/3 日分别预测
    result = latest_df[[TS_COL] + [c for c in GROUP_KEYS if c in latest_df.columns]].copy()
    result = result.reset_index(drop=True)

    for h in [1, 2, 3]:
        model_path = MODEL_ROOT / f"model_{h}d.pkl"
        if not model_path.exists():
            print(f"⚠️ 未找到模型文件：{model_path}，跳过 {h} 日预测。")
            continue

        print(f"▶ 使用 {h} 日模型做未来预测：{model_path.name}")
        model = joblib.load(model_path)
        y_pred = model.predict(X)

        result[f"pred_price_{h}d"] = y_pred

        # 预测日期（当前 ts + h 天）
        result[f"pred_ts_{h}d"] = result[TS_COL] + pd.to_timedelta(h, unit="D")

    # 5. 保存结果
    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n💾 已输出未来预测结果：{OUTPUT_CSV}")
    print("   主要字段：ts(当前)、pred_ts_*d(预测日期)、pred_price_*d(预测价格)")


if __name__ == "__main__":
    main()
