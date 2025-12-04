# src/prediction_models/short_term_price_pred/model_train.py
"""
短期价格预测模型训练脚本（时间序列切分版）
A 步：1/2/3 天价格预测模型训练

更新内容：
✔ 取消随机切分（train_test_split）
✔ 改为真正的时间序列切分（前 80% 训练，后 20% 验证）
✔ 特征编码使用 category.codes，避免 dtype 错误
✔ 生成与保存模型 + 元数据

使用方式：
    python -m src.prediction_models.short_term_price_pred.model_train
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------
# 路径与常量
# -------------------------

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
FEATURE_DIR = DATA_DIR / "intermediate" / "features"
FEATURE_CSV = FEATURE_DIR / "time_series_features.csv"

MODEL_ROOT = ROOT / "models" / "artifacts" / "price_model_v1"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

TS_COL = "ts"
TRAIN_FRACTION = 0.8
RANDOM_STATE = 42


@dataclass
class ModelTrainReport:
    horizon: int
    target_col: str
    n_train: int
    n_valid: int
    train_date_range: str
    valid_date_range: str
    mae: float
    rmse: float
    mape: float
    feature_count: int
    categorical_features: List[str]
    model_params: Dict


# -------------------------
# 数据加载
# -------------------------

def load_feature_data() -> pd.DataFrame:
    """读取特征工程输出文件"""
    if not FEATURE_CSV.exists():
        raise FileNotFoundError(f"特征文件不存在：{FEATURE_CSV}")
    df = pd.read_csv(FEATURE_CSV)
    df[TS_COL] = pd.to_datetime(df[TS_COL])
    return df


# -------------------------
# 时间切分构建训练集/验证集
# -------------------------

def build_train_valid_split(
    df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str], pd.Index, pd.Index]:

    # 删除没有目标值的行
    df = df.dropna(subset=[target_col]).copy()

    # 按时间排序
    df = df.sort_values(TS_COL).reset_index(drop=True)

    # 计算时间切分点
    n_total = len(df)
    split_idx = int(n_total * TRAIN_FRACTION)

    df_train = df.iloc[:split_idx].copy()
    df_valid = df.iloc[split_idx:].copy()

    # 目标列
    target_cols = [c for c in df.columns if c.startswith("y_price_") or c.startswith("y_volume_")]
    drop_cols = set(target_cols + [TS_COL])

    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_train = df_train[feature_cols].copy()
    y_train = df_train[target_col]

    X_valid = df_valid[feature_cols].copy()
    y_valid = df_valid[target_col]

    # 找类别列（object）
    categorical_cols = [c for c in feature_cols if X_train[c].dtype == "object"]

    # 类别编码（不泄漏未来）
    for col in categorical_cols:
        X_train[col] = X_train[col].astype("category").cat.codes.astype("int32")
        X_valid[col] = X_valid[col].astype("category").cat.codes.astype("int32")

    return (
        X_train,
        X_valid,
        y_train,
        y_valid,
        feature_cols,
        categorical_cols,
        df_train.index,
        df_valid.index,
    )


# -------------------------
# 模型训练
# -------------------------

def train_single_horizon_model(horizon: int) -> ModelTrainReport:
    target_col = f"y_price_{horizon}d"

    print("\n======================")
    print(f"▶ 开始训练 {horizon} 日模型（target={target_col}）")
    print("======================")

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

    # LightGBM 参数
    model_params = {
        "objective": "regression",
        "n_estimators": 600,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.2,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    model = LGBMRegressor(**model_params)

    # 训练模型
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l1",
    )

    # 验证集预测
    y_pred = model.predict(X_valid)

    mae = float(mean_absolute_error(y_valid, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred)))
    mape = float(np.mean(np.abs((y_valid - y_pred) / (y_valid + 1e-6))))

    # 日期范围
    train_range = f"{df.loc[train_idx, TS_COL].min()} ~ {df.loc[train_idx, TS_COL].max()}"
    valid_range = f"{df.loc[valid_idx, TS_COL].min()} ~ {df.loc[valid_idx, TS_COL].max()}"

    print(f"📊 {horizon} 日验证集：MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.2%}")

    # 保存模型
    model_path = MODEL_ROOT / f"model_{horizon}d.pkl"
    metadata_path = MODEL_ROOT / f"metadata_{horizon}d.json"

    joblib.dump(model, model_path)

    report = ModelTrainReport(
        horizon=horizon,
        target_col=target_col,
        n_train=len(X_train),
        n_valid=len(X_valid),
        train_date_range=train_range,
        valid_date_range=valid_range,
        mae=mae,
        rmse=rmse,
        mape=mape,
        feature_count=len(feature_cols),
        categorical_features=categorical_cols,
        model_params=model_params,
    )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)

    # 同步保存默认模型（1 日）
    if horizon == 1:
        joblib.dump(model, MODEL_ROOT / "model.pkl")
        with open(MODEL_ROOT / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)

    return report


# -------------------------
# 入口
# -------------------------

def main():
    print("🌼 开始训练短期价格预测模型（时间序列切分版）...")
    reports: List[ModelTrainReport] = []

    for h in [1, 2, 3]:
        try:
            r = train_single_horizon_model(h)
            reports.append(r)
        except KeyError:
            print(f"⚠️ 数据中缺少 y_price_{h}d，跳过。")

    print("\n======================")
    print("✅ 所有模型训练完成（时间切分版）")
    print("======================")

    for r in reports:
        print(
            f"- {r.horizon} 日：MAE={r.mae:.4f} RMSE={r.rmse:.4f} MAPE={r.mape:.2%} "
            f"训练样本={r.n_train:,} 验证样本={r.n_valid:,}"
        )


if __name__ == "__main__":
    main()
