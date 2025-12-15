# src/prediction_models/common/inject_pred_price_feature.py
# -*- coding: utf-8 -*-

"""
将短期价格预测结果注入到时间序列特征表中：

- 读取：
    data/intermediate/features/time_series_features.csv
    data/output/price_prediction_wide.csv
- 合并键：
    ts + product_id
- 新增内容：
    pred_price_1d / 2d / 3d
    has_pred_price：是否存在真实的价格预测（主要依据 pred_price_1d）
- 对缺失的预测价格进行 fallback：
    使用当日 retail_price 进行填补（若仍为 NaN 则保留）
"""

from pathlib import Path
import pandas as pd


# 路径定义：相对项目根目录
FEATURE_PATH = Path("data/intermediate/features/time_series_features.csv")
BACKUP_PATH = Path("data/intermediate/features/time_series_features_backup.csv")
PRICE_PRED_PATH = Path("data/output/price_prediction_wide.csv")

# 合并键（与价格预测宽表保持一致）
JOIN_KEYS = ["ts", "product_id"]

REQUIRED_PRICE_COLS = ["pred_price_1d", "pred_price_2d", "pred_price_3d"]


def _safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _safe_to_int(s: pd.Series) -> pd.Series:
    # 有的 CSV 会把 id 读成 float，这里统一转成 Int64
    return s.astype("Int64")


def main():
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"❌ 特征文件不存在：{FEATURE_PATH}")
    if not PRICE_PRED_PATH.exists():
        raise FileNotFoundError(f"❌ 价格预测结果不存在：{PRICE_PRED_PATH}")

    print(f"📥 读取特征数据：{FEATURE_PATH}")
    df_feat = pd.read_csv(FEATURE_PATH)

    print(f"📥 读取价格预测结果（宽格式）：{PRICE_PRED_PATH}")
    df_price = pd.read_csv(PRICE_PRED_PATH)

    # ---- 检查必备列 ----
    for col in JOIN_KEYS:
        if col not in df_feat.columns:
            raise KeyError(f"特征文件缺少合并键列：{col}")
        if col not in df_price.columns:
            raise KeyError(f"价格预测文件缺少合并键列：{col}")

    for col in REQUIRED_PRICE_COLS:
        if col not in df_price.columns:
            raise KeyError(f"价格预测文件缺少列：{col}")

    # ---- 对齐类型：ts + product_id ----
    print("🧩 对齐合并键的数据类型（ts -> datetime, product_id -> Int64） ...")
    # 特征表
    df_feat["ts"] = _safe_to_datetime(df_feat["ts"])
    df_feat["product_id"] = _safe_to_int(df_feat["product_id"])

    # 价格预测表
    df_price["ts"] = _safe_to_datetime(df_price["ts"])
    df_price["product_id"] = _safe_to_int(df_price["product_id"])

    # ---- 只保留需要的列参与合并 ----
    df_price_small = df_price[JOIN_KEYS + REQUIRED_PRICE_COLS].copy()

    # ---- 执行合并 ----
    print(f"🔗 使用 join key 合并：{JOIN_KEYS}")
    df_merged = df_feat.merge(
        df_price_small,
        on=JOIN_KEYS,
        how="left",
        suffixes=("", "_predtmp"),
    )

    # 如果之前已经有 pred_price_xd，先删掉旧的（避免重复列）
    for col in REQUIRED_PRICE_COLS:
        if col in df_feat.columns and col in df_merged.columns:
            df_merged.drop(columns=[col], inplace=True)

    # 处理 merge 后可能产生的 *_predtmp 列
    for col in REQUIRED_PRICE_COLS:
        alt = f"{col}_predtmp"
        if alt in df_merged.columns and col not in df_merged.columns:
            df_merged.rename(columns={alt: col}, inplace=True)

    # ---- 统计注入覆盖率 ----
    total_rows = len(df_merged)
    print(f"📊 合并后总行数：{total_rows:,}")

    for col in REQUIRED_PRICE_COLS:
        if col in df_merged.columns:
            filled_ratio = df_merged[col].notna().mean() * 100
            print(f"   {col} 注入成功比例：{filled_ratio:.2f}%")
        else:
            print(f"   ⚠ 未找到列 {col}（可能合并失败或被重命名）")

    # ---- 新增 has_pred_price 标记列 ----
    # 只要 1 日预测存在，就认为该样本有“真实价格预测”
    if "pred_price_1d" in df_merged.columns:
        df_merged["has_pred_price"] = df_merged["pred_price_1d"].notna().astype(int)
    else:
        # 极端兜底：如果列不存在，全部置 0
        df_merged["has_pred_price"] = 0
        print("⚠ 未找到 pred_price_1d 列，has_pred_price 全部为 0。")

    # ---- 对缺失的预测价格做 fallback（用当日零售价代替） ----
    if "retail_price" in df_merged.columns:
        for col in REQUIRED_PRICE_COLS:
            if col in df_merged.columns:
                before_na = df_merged[col].isna().sum()
                df_merged[col] = df_merged[col].fillna(df_merged["retail_price"])
                after_na = df_merged[col].isna().sum()
                print(
                    f"   {col} 使用 retail_price 填补 NaN：{before_na} -> {after_na} 个缺失"
                )
    else:
        print("⚠ 特征表中不存在 retail_price 列，无法对缺失预测价格做 fallback。")

    # ---- 备份原始特征文件 ----
    print(f"💾 备份原始特征文件到：{BACKUP_PATH}")
    BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(BACKUP_PATH, index=False)

    # ---- 写回带有预测价格特征的新特征文件 ----
    print(f"💾 写回带预测价格特征的特征文件：{FEATURE_PATH}")
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(FEATURE_PATH, index=False)

    print("✅ 预测价格特征注入完成。")


if __name__ == "__main__":
    main()
