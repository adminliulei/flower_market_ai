# src/prediction_models/volume_pred/model_train_A.py
# -*- coding: utf-8 -*-

"""
方案 A：使用“预测价格 + has_pred_price 标记”的成交量预测模型训练脚本

核心变化：
- 不再丢弃 pred_price_1d 为空的样本；
- 若注入阶段仍有缺失，使用当日 retail_price 作为 fallback；
- 新增特征 has_pred_price，帮助模型区分“真实预测价”和“fallback 价”。
"""

from pathlib import Path
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


FEATURE_PATH = Path("data/intermediate/features/time_series_features.csv")
MODEL_DIR = Path("models/artifacts/volume_model_A/")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


PRED_PRICE_COLS = ["pred_price_1d", "pred_price_2d", "pred_price_3d"]


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURE_PATH)

    # ts 转 datetime，后面按时间切分
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    # 若注入阶段尚未写入 has_pred_price，这里兜底生成一列
    if "has_pred_price" not in df.columns:
        df["has_pred_price"] = df["pred_price_1d"].notna().astype(int)

    # 对预测价格再次做 fallback（双保险）
    if "retail_price" in df.columns:
        for col in PRED_PRICE_COLS:
            if col in df.columns and df[col].isna().any():
                before_na = df[col].isna().sum()
                df[col] = df[col].fillna(df["retail_price"])
                after_na = df[col].isna().sum()
                print(
                    f"[train] {col} 使用 retail_price 再次填补 NaN：{before_na} -> {after_na}"
                )

    return df


def train_single_model(horizon: int, df: pd.DataFrame):
    target = f"y_volume_{horizon}d"

    # -------- 去掉标签缺失的行（shift 产生的尾部 NaN 等）--------
    df = df[~df[target].isna()].copy()

    # -------- 特征选择：去掉标签 & ts --------
    drop_cols = [
        "ts",
        # 所有 y 标签
        "y_price_1d",
        "y_volume_1d",
        "y_price_2d",
        "y_volume_2d",
        "y_price_3d",
        "y_volume_3d",
        target,  # 当前 horizon 的目标列
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]

    # log1p 处理目标（成交量长尾）
    df[target] = np.log1p(df[target].clip(lower=0))

    # -------- 在整个 df 上先把类别列统一转成 category --------
    obj_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        df[col] = df[col].astype("category")

    # -------- 按时间切分（更接近真实场景） --------
    df = df.sort_values("ts")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    valid_df = df.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols], train_df[target]
    X_valid, y_valid = valid_df[feature_cols], valid_df[target]

    print(
        f"[H{horizon}] 训练集行数={len(train_df):,}，验证集行数={len(valid_df):,}，特征数={len(feature_cols)}"
    )

    # -------- LightGBM 模型 --------
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.7,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50)],
    )

    # -------- 保存模型 --------
    model_path = MODEL_DIR / f"model_{horizon}d.pkl"
    joblib.dump(model, model_path)

    # -------- 保存元数据 --------
    meta = {
        "horizon": horizon,
        "features": feature_cols,
        "categorical_features": obj_cols,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
    }
    with open(MODEL_DIR / f"metadata_{horizon}d.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"✅ 方案 A：第 {horizon} 天成交量模型训练完成 → {model_path} "
        f"(train={len(train_df):,}, valid={len(valid_df):,})"
    )


def main():
    df = load_features()
    print(f"📊 可用于方案 A 的训练样本数（不再过滤 pred_price_1d）：{len(df):,}")

    for h in [1, 2, 3]:
        train_single_model(h, df)


if __name__ == "__main__":
    main()
