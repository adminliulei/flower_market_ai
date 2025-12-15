# src/prediction_models/short_term_price_pred/model_predict_A_old.py
"""
短期价格预测 - 验证集回测预测脚本

功能：
1. 读取特征工程输出（time_series_features.csv）
2. 使用 model_train_A_old.py 中的时间序列切分逻辑获取验证集
3. 分别加载 1/2/3 日模型，在验证集上做预测
4. 输出：data/output/price_prediction_result.csv
    - ts, product_id, variety, ..., horizon, y_true, y_pred, error, abs_error, ape(%)

运行方式（项目根目录）：
    python -m src.prediction_models.short_term_price_pred.model_predict
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

from .model_train import (
    load_feature_data,
    build_train_valid_split,
    MODEL_ROOT,
    TS_COL,
)


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "price_prediction_result.csv"


def predict_for_horizon(horizon: int) -> pd.DataFrame:
    """
    使用已训练好的 {horizon} 日模型，在**验证集**上生成预测结果。

    返回列：
        ts, horizon, product_id, variety, classify_name, color,
        grade, spec, shop_name, y_true, y_pred, error, abs_error, ape
    """
    target_col = f"y_price_{horizon}d"
    model_path = MODEL_ROOT / f"model_{horizon}d.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件：{model_path}")

    print("\n======================")
    print(f"▶ 使用 {horizon} 日模型做验证集回测：{model_path.name}")
    print("======================")

    # 1. 加载特征
    df = load_feature_data()

    (
        X_train,
        X_valid,
        y_train,
        y_valid,
        feature_cols,
        categorical_cols,
        train_idx,
        valid_idx,
    ) = build_train_valid_split(df, target_col)

    # 2. 加载模型
    model = joblib.load(model_path)

    # 3. 验证集预测
    y_pred = model.predict(X_valid)

    eps = 1e-6
    error = y_pred - y_valid.values
    abs_error = np.abs(error)
    ape = abs_error / (np.abs(y_valid.values) + eps) * 100.0

    # 4. 组织结果
    base_cols: List[str] = [
        TS_COL,
        "product_id",
        "variety",
        "classify_name",
        "color",
        "grade",
        "spec",
        "shop_name",
    ]
    existing_base_cols = [c for c in base_cols if c in df.columns]

    result_df = df.loc[valid_idx, existing_base_cols].copy()
    result_df["horizon"] = horizon
    result_df["target_col"] = target_col
    result_df["y_true"] = y_valid.values
    result_df["y_pred"] = y_pred
    result_df["error"] = error
    result_df["abs_error"] = abs_error
    result_df["ape"] = ape

    result_df = result_df.sort_values([TS_COL, "horizon"]).reset_index(drop=True)

    return result_df


def main():
    print("🌼 短期价格预测 - 验证集回测开始 ...")

    all_results: List[pd.DataFrame] = []

    for h in [1, 2, 3]:
        try:
            df_h = predict_for_horizon(h)
            all_results.append(df_h)
        except FileNotFoundError as e:
            print(f"⚠️ 跳过 {h} 日模型：{e}")

    if not all_results:
        print("❌ 未生成任何预测结果，请检查模型文件是否存在。")
        return

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n💾 已写出预测结果 CSV：{OUTPUT_CSV}")
    print("   字段：ts, product_id, ..., horizon, y_true, y_pred, error, abs_error, ape(%)")


if __name__ == "__main__":
    main()
