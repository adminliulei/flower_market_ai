# src/prediction_models/volume_pred/model_predict_B.py

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


FEATURE_PATH = Path("data/intermediate/features/time_series_features.csv")
MODEL_DIR = Path("models/artifacts/volume_model_B")
OUTPUT_PATH = Path("data/output/volume_prediction_result_B.csv")


def load_features() -> pd.DataFrame:
    """加载特征数据（方案 B：不用预测价格，仅使用完整特征表）"""
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"特征文件不存在：{FEATURE_PATH}")
    df = pd.read_csv(FEATURE_PATH)

    # 确保 ts 为 datetime，便于时间切分
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    return df


def load_metadata(horizon: int) -> dict:
    meta_path = MODEL_DIR / f"metadata_{horizon}d.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"未找到元数据文件：{meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(horizon: int):
    model_path = MODEL_DIR / f"model_{horizon}d.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件：{model_path}")
    return joblib.load(model_path)


def main():
    print("📥 读取特征数据 ...")
    df = load_features()
    df = df.sort_values("ts").reset_index(drop=True)

    n_total = len(df)
    split_idx = int(n_total * 0.8)
    valid_df = df.iloc[split_idx:].copy()
    print(f"📌 使用后 20% 数据作为验证集进行预测：valid_rows={len(valid_df):,}")

    # 标识列（方便后续对接业务或评估）
    id_cols_candidate = [
        "ts",
        "product_id",
        "variety",
        "classify_name",
        "shop_name",
        "spec",
        "grade",
        "color",
        "market_name",
        "place",
    ]
    id_cols = [c for c in id_cols_candidate if c in valid_df.columns]

    # 结果表：先放 id 列
    result = valid_df[id_cols].copy() if id_cols else valid_df[["ts"]].copy()

    # 顺便把真实值列也一并写出（方便后续评估）
    for target_col in ["y_volume_1d", "y_volume_2d", "y_volume_3d"]:
        if target_col in valid_df.columns:
            # 原始 CSV 中是原始成交量，不需要 expm1，剪掉负值即可
            result[target_col] = valid_df[target_col].clip(lower=0)

    # 依次加载 1/2/3 天的模型做预测
    for horizon in [1, 2, 3]:
        target_col = f"y_volume_{horizon}d"
        meta = load_metadata(horizon)
        feature_cols = meta["features"]
        cat_cols = meta.get("categorical_features", []) or []

        # 检查特征列是否齐全
        missing_cols = [c for c in feature_cols if c not in valid_df.columns]
        if missing_cols:
            raise ValueError(f"h={horizon}d 缺少特征列：{missing_cols}")

        # 按训练时一致的方式处理类别列
        for col in cat_cols:
            if col in valid_df.columns:
                valid_df[col] = valid_df[col].astype("category")

        model = load_model(horizon)
        X_valid = valid_df[feature_cols]

        print(f"▶ 预测 {horizon} 日成交量（方案 B，仅历史价格） ...")
        # 模型输出的是 log1p(volume)，这里反变换回原始成交量
        y_pred_log = model.predict(X_valid)
        y_pred = np.expm1(y_pred_log)
        # 防止溢出 / inf / nan
        y_pred = np.where(np.isfinite(y_pred), y_pred, np.nan)
        # 不允许负成交量
        y_pred = np.maximum(y_pred, 0)

        pred_col = f"pred_volume_{horizon}d_B"
        result[pred_col] = y_pred

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ 方案 B 预测结果已写入：{OUTPUT_PATH}")
    print("   字段示例：ts, product_id, variety, ..., y_volume_1d/2d/3d, pred_volume_1d/2d/3d_B")


if __name__ == "__main__":
    main()
