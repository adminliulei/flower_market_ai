# src/prediction_models/volume_pred/model_predict_A.py
# -*- coding: utf-8 -*-

"""
方案 A：使用“预测价格 + has_pred_price 标记”的成交量预测模型预测脚本

- 使用与训练完全一致的特征构造方式（包括 fallback 逻辑）；
- 对验证集（后 20% 时间段）进行预测；
- 输出竖表结果：每行一个 ts × product × horizon。
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


FEATURE_PATH = Path("data/intermediate/features/time_series_features.csv")
MODEL_DIR = Path("models/artifacts/volume_model_A/")
OUTPUT_PATH = Path("data/output/volume_prediction_result_A.csv")

PRED_PRICE_COLS = ["pred_price_1d", "pred_price_2d", "pred_price_3d"]


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURE_PATH)

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    # 若不存在 has_pred_price，则兜底生成
    if "has_pred_price" not in df.columns:
        df["has_pred_price"] = df["pred_price_1d"].notna().astype(int)

    # 对预测价格再次做 fallback（与训练侧保持一致）
    if "retail_price" in df.columns:
        for col in PRED_PRICE_COLS:
            if col in df.columns and df[col].isna().any():
                before_na = df[col].isna().sum()
                df[col] = df[col].fillna(df["retail_price"])
                after_na = df[col].isna().sum()
                print(
                    f"[predict] {col} 使用 retail_price 再次填补 NaN：{before_na} -> {after_na}"
                )

    return df


def predict_single_horizon(horizon: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    使用已训练好的方案 A 成交量模型，对验证集（后 20%）做预测。
    """
    target = f"y_volume_{horizon}d"

    # ---- 读取模型与元数据 ----
    model_path = MODEL_DIR / f"model_{horizon}d.pkl"
    meta_path = MODEL_DIR / f"metadata_{horizon}d.json"

    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在：{model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"元数据不存在：{meta_path}")

    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = meta["features"]
    cat_cols = meta.get("categorical_features", [])

    # ---- 去掉标签缺失的行（与训练保持一致）----
    df = df[~df[target].isna()].copy()

    # ---- 按时间排序并切分：后 20% 作为验证集/评估集 ----
    df_sorted = df.sort_values("ts").copy()
    split_idx = int(len(df_sorted) * 0.8)
    valid_df = df_sorted.iloc[split_idx:].copy()

    print(f"📌 使用后 20% 数据作为验证集进行预测：valid_rows={len(valid_df):,}")

    # ---- 处理类别列：和训练时保持一致 → category ----
    for col in cat_cols:
        if col in valid_df.columns:
            valid_df[col] = valid_df[col].astype("category")

    # ---- 确保特征列在 df 中存在 ----
    missing = [c for c in feature_cols if c not in valid_df.columns]
    if missing:
        raise KeyError(f"特征列在特征文件中缺失：{missing}")

    X_valid = valid_df[feature_cols]

    # ---- 真实值：CSV 里是原始成交量，不需要 expm1 ----
    if target not in valid_df.columns:
        raise KeyError(f"目标列缺失：{target}")
    y_true = valid_df[target].clip(lower=0)

    # ---- 预测（模型输出的是 log1p 后的量）---->
    print(f"▶ 预测 {horizon} 日成交量（方案 A，使用预测价格） ...")
    y_pred_log = model.predict(X_valid)
    # 预测值是 log1p(volume)，这里反变换回原始成交量
    y_pred = np.expm1(y_pred_log)
    # 防止极端值导致 inf / overflow
    y_pred = np.where(np.isfinite(y_pred), y_pred, np.nan)
    # 不允许负成交量
    y_pred = np.maximum(y_pred, 0)

    # ---- 组装结果 ----
    out_cols_base = [
        "ts",
        "product_id",
        "variety",
        "grade",
        "market_name",
        "classify_name",
        "spec",
        "color",
        "place",
        "shop_name",
        "has_pred_price",  # 方便后续分析：哪些样本真的有价格预测
    ]
    out_cols_exist = [c for c in out_cols_base if c in valid_df.columns]

    result = valid_df[out_cols_exist].copy()
    result["horizon"] = horizon
    result["y_true"] = y_true
    result["y_pred"] = y_pred
    result["abs_error"] = (result["y_pred"] - result["y_true"]).abs()
    result["ape"] = np.where(
        result["y_true"] > 0,
        result["abs_error"] / result["y_true"] * 100,
        np.nan,
    )

    return result


def main():
    print("📥 读取特征数据 ...")
    df = load_features()

    all_results = []
    for h in [1, 2, 3]:
        res_h = predict_single_horizon(h, df)
        all_results.append(res_h)

    df_out = pd.concat(all_results, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"💾 已写出方案 A 成交量预测结果：{OUTPUT_PATH}")
    print(
        "   字段：ts, product_id, variety, ..., has_pred_price, horizon, "
        "y_true, y_pred, abs_error, ape(%)"
    )


if __name__ == "__main__":
    main()
