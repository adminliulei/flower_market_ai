# src/prediction_models/volume_pred/model_train_B.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import json
import joblib

FEATURE_PATH = Path("data/intermediate/features/time_series_features.csv")
MODEL_DIR = Path("models/artifacts/volume_model_B/")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURE_PATH)

    # 方案 B：不用预测价格 → 直接丢掉 pred_price_* 列（即使有也不用）
    for col in ["pred_price_1d", "pred_price_2d", "pred_price_3d"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # ts 转 datetime，后面按时间切分
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    return df


def train_single_model(horizon: int, df: pd.DataFrame):
    target = f"y_volume_{horizon}d"

    # -------- 特征选择：去掉标签 & ts --------
    drop_cols = [
        "ts",
        # 所有 y 标签
        "y_price_1d", "y_volume_1d",
        "y_price_2d", "y_volume_2d",
        "y_price_3d", "y_volume_3d",
        target,  # 当前 horizon 的目标列
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]

    # log1p 处理目标（成交量长尾）
    df = df.copy()
    df[target] = np.log1p(df[target])

    # -------- 在整个 df 上把类别列统一转成 category --------
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

    # -------- LightGBM 模型 --------
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.7,
        colsample_bytree=0.8,
    )

    model.fit(
        X_train, y_train,
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
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
    }
    with open(MODEL_DIR / f"metadata_{horizon}d.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"✅ 方案 B：第 {horizon} 天成交量模型训练完成 → {model_path} "
        f"(train={len(train_df):,}, valid={len(valid_df):,})"
    )


def main():
    df = load_features()
    print(f"📊 方案 B 可用训练样本数：{len(df):,}")

    for h in [1, 2, 3]:
        train_single_model(h, df)


if __name__ == "__main__":
    main()
