# src/business_decision/procurement_core.py
# -*- coding: utf-8 -*-
"""
F 步：采购建议核心逻辑（增强版）

功能：
- 从成交量预测 / 价格预测生成采购建议
- 支持 0 库存或从 DB 获取库存
- 考虑安全系数、价格信号、保鲜期（库存覆盖天数上限）
- 支持按预算自动收缩采购量
"""

from __future__ import annotations

from pathlib import Path
import math
from datetime import datetime
from typing import Optional, Dict

import numpy as pd
import pandas as pd

# 尽量复用你现有的工具
try:
    from src.utils.db_utils import get_local_conn  # C1 / D 步里用过
except ImportError:
    get_local_conn = None

# ---------- 路径配置 ----------
VOLUME_PRED_PATH = Path("data/output/volume_prediction_result_A.csv")
PRICE_PRED_PATH = Path("data/output/price_prediction_result.csv")
OUTPUT_PATH = Path("data/output/procurement_suggestion.csv")

# 没有库存表时直接用 0 库存
INVENTORY_FROM_DB = False

# ---------- 参数配置（可后续挪到 config/settings.py） ----------

MAX_HORIZON_DAYS = 3           # 需求视野：未来 3 天
SAFETY_FACTOR_BASE = 0.2       # 基础安全系数 20%
HIGH_PRICE_TH = 1.05           # 价格偏高阈值
LOW_PRICE_TH = 0.95            # 价格偏低阈值

# 日度采购预算（元），0 或 None 表示不启用预算约束
DAILY_BUDGET: Optional[float] = 300000.0

# 简单保鲜期配置：按品类控制最大库存覆盖天数
# 你可以根据真实业务调这个字典
SHELF_LIFE_CONFIG: Dict[str, int] = {
    "单头玫瑰": 3,
    "多头玫瑰": 3,
    "单头康乃馨": 4,
    "菊花类": 5,
}
DEFAULT_MAX_COVERAGE_DAYS = 5


# ======================= 数据加载 =======================

def load_volume_forecast() -> pd.DataFrame:
    """
    读取成交量预测结果（方案 A），长表 → 聚合出未来 1/3 天需求。

    volume_prediction_result_A.csv 结构示例：
    ts, product_id, variety, grade, market_name, classify_name,
    spec, color, place, shop_name, has_pred_price, horizon, y_true, y_pred, ...
    """
    if not VOLUME_PRED_PATH.exists():
        raise FileNotFoundError(f"找不到成交量预测结果文件：{VOLUME_PRED_PATH}")

    df = pd.read_csv(VOLUME_PRED_PATH)

    df = df[df["horizon"].isin([1, 2, 3])]

    key_cols = [
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
    ]

    df_h1 = (
        df[df["horizon"] == 1]
        .groupby(key_cols, as_index=False)["y_pred"]
        .sum()
        .rename(columns={"y_pred": "y_volume_1d_pred"})
    )

    df_h123 = (
        df[df["horizon"].isin([1, 2, 3])]
        .groupby(key_cols, as_index=False)["y_pred"]
        .sum()
        .rename(columns={"y_pred": "y_volume_3d_pred"})
    )

    df_merged = df_h1.merge(df_h123, on=key_cols, how="left")
    df_merged["y_volume_1d_pred"] = df_merged["y_volume_1d_pred"].clip(lower=0)
    df_merged["y_volume_3d_pred"] = df_merged["y_volume_3d_pred"].clip(lower=0)

    return df_merged


def load_price_forecast() -> Optional[pd.DataFrame]:
    """
    读取价格预测结果（horizon=1），自适应列名。
    price_prediction_result.csv 结构示例（根据你截图）：
    ts, product_id, variety, classify_name, color, grade, spec, shop_name,
    horizon, target_col, y_true, y_pred, ...
    """
    if not PRICE_PRED_PATH.exists():
        print(f"⚠ 未找到价格预测文件：{PRICE_PRED_PATH}；将跳过价格信号与预算约束。")
        return None

    df = pd.read_csv(PRICE_PRED_PATH)
    if "horizon" in df.columns:
        df = df[df["horizon"] == 1].copy()

    price_cols = [
        "ts",
        "product_id",
        "variety",
        "classify_name",
        "color",
        "grade",
        "spec",
        "shop_name",
    ]
    key_cols = [c for c in price_cols if c in df.columns]

    if "y_pred" not in df.columns:
        raise ValueError("价格预测文件缺少 y_pred 列！")

    df_price = (
        df.groupby(key_cols, as_index=False)["y_pred"]
        .mean()
        .rename(columns={"y_pred": "y_price_1d_pred"})
    )
    return df_price


def load_inventory_from_db(run_date: datetime.date) -> pd.DataFrame:
    """
    从 PostgreSQL 读取某天的库存快照。
    你将来在 DB 里建好 fm_inventory_daily_snapshot 表后即可启用。
    """
    if get_local_conn is None:
        return pd.DataFrame()

    sql = """
    SELECT
        snapshot_date AS run_date,
        product_id,
        variety,
        grade,
        market_name,
        classify_name,
        spec,
        color,
        place,
        shop_name,
        stock_on_hand,
        stock_in_transit
    FROM fm_inventory_daily_snapshot
    WHERE snapshot_date = %(run_date)s
    """
    with get_local_conn() as conn:
        df = pd.read_sql(sql, conn, params={"run_date": run_date})
    return df


def make_zero_inventory(df_vol: pd.DataFrame, run_date: datetime.date) -> pd.DataFrame:
    """在没有库存数据时，生成全 0 库存快照。"""
    inv_cols = [
        "run_date",
        "product_id",
        "variety",
        "grade",
        "market_name",
        "classify_name",
        "spec",
        "color",
        "place",
        "shop_name",
        "stock_on_hand",
        "stock_in_transit",
    ]
    df_key = (
        df_vol[
            [
                "product_id",
                "variety",
                "grade",
                "market_name",
                "classify_name",
                "spec",
                "color",
                "place",
                "shop_name",
            ]
        ]
        .drop_duplicates()
        .copy()
    )
    df_key["run_date"] = run_date
    df_key["stock_on_hand"] = 0.0
    df_key["stock_in_transit"] = 0.0

    return df_key[inv_cols]


# ======================= 采购逻辑 =======================

def _get_max_coverage_days(classify_name: str) -> int:
    """根据品类获取允许的最大库存覆盖天数（保鲜期约束）。"""
    if isinstance(classify_name, str):
        for key, days in SHELF_LIFE_CONFIG.items():
            if key in classify_name:
                return days
    return DEFAULT_MAX_COVERAGE_DAYS


def compute_procurement_for_row(row: pd.Series) -> pd.Series:
    """
    对单 SKU 计算采购建议。
    需要 row 包含：
    - y_volume_1d_pred, y_volume_3d_pred
    - stock_on_hand, stock_in_transit
    - classify_name（可选）
    - y_price_1d_pred（可选）
    """
    # 1. 需求视角：未来 1/3 天
    v1 = max(row.get("y_volume_1d_pred", 0), 0)
    v3 = max(row.get("y_volume_3d_pred", 0), 0)

    forecast_demand_1d = v1
    forecast_demand_3d = v3
    avg_daily_demand = forecast_demand_3d / MAX_HORIZON_DAYS if MAX_HORIZON_DAYS > 0 else 0

    # 2. 安全系数
    required_base = forecast_demand_3d
    required_with_safety = required_base * (1 + SAFETY_FACTOR_BASE)

    # 3. 价格信号（目前暂不使用 recent_avg_price，后续可接 E 步指数）
    price_signal = "normal"
    price_factor = 1.0
    y_price_1d_pred = row.get("y_price_1d_pred", None)

    # 占位逻辑：如果以后你给 recent_avg_price，就能启用价格调节
    recent_avg_price = row.get("recent_avg_price", None)
    if pd.notna(y_price_1d_pred) and pd.notna(recent_avg_price) and recent_avg_price > 0:
        price_ratio = y_price_1d_pred / recent_avg_price
        if price_ratio <= LOW_PRICE_TH:
            price_signal = "low_price_buy_more"
            price_factor = 1.1
        elif price_ratio >= HIGH_PRICE_TH:
            price_signal = "high_price_cautious"
            price_factor = 0.9

    required_stock_level = required_with_safety * price_factor

    # 4. 扣减库存
    stock_on_hand = float(row.get("stock_on_hand", 0) or 0)
    stock_in_transit = float(row.get("stock_in_transit", 0) or 0)
    available_future_stock = stock_on_hand + stock_in_transit

    purchase_qty_raw = required_stock_level - available_future_stock
    recommended_purchase_qty = max(purchase_qty_raw, 0.0)

    # 5. 保鲜期约束：限制覆盖天数
    classify_name = row.get("classify_name", "")
    max_days = _get_max_coverage_days(str(classify_name))
    if avg_daily_demand > 0:
        # 按当前推荐采购量计算覆盖天数
        coverage_days = (available_future_stock + recommended_purchase_qty) / avg_daily_demand
        if coverage_days > max_days:
            max_allowed_stock = avg_daily_demand * max_days
            max_allowed_purchase = max(0.0, max_allowed_stock - available_future_stock)
            recommended_purchase_qty = min(recommended_purchase_qty, max_allowed_purchase)

    # 6. 取整到最小采购单位（这里先用 1，未来可加 min_pack_size）
    if recommended_purchase_qty > 0:
        purchase_qty_rounded = math.ceil(recommended_purchase_qty)
    else:
        purchase_qty_rounded = 0

    # 7. 决策说明
    if purchase_qty_rounded == 0:
        decision_reason = "库存+在途已覆盖未来需求或需求过低，无需采购"
    else:
        decision_reason = f"按未来3天需求、安全系数与保鲜期约束计算，建议采购 {purchase_qty_rounded} 单位"

    row["forecast_demand_1d"] = forecast_demand_1d
    row["forecast_demand_3d"] = forecast_demand_3d
    row["avg_daily_demand"] = avg_daily_demand
    row["required_stock_level"] = required_stock_level
    row["recommended_purchase_qty"] = recommended_purchase_qty
    row["purchase_qty_rounded"] = purchase_qty_rounded
    row["price_signal"] = price_signal
    row["decision_reason"] = decision_reason

    return row


def apply_budget_constraint(df: pd.DataFrame, daily_budget: Optional[float]) -> pd.DataFrame:
    """
    按日预算约束采购总额：
    - 使用 y_price_1d_pred * purchase_qty_rounded 估算成本
    - 若总成本 > 预算，则按“缺货风险优先”缩减采购量
    """
    if not daily_budget or daily_budget <= 0:
        return df

    if "y_price_1d_pred" not in df.columns:
        print("⚠ 未找到 y_price_1d_pred，无法做预算约束，跳过。")
        return df

    df = df.copy()
    df["est_cost"] = df["y_price_1d_pred"].fillna(0) * df["purchase_qty_rounded"]
    total_cost = df["est_cost"].sum()

    if total_cost <= daily_budget:
        return df

    print(f"⚠ 估算采购金额 {total_cost:,.0f} 超出预算 {daily_budget:,.0f}，将按优先级缩减采购量。")

    # 优先级：需求大且库存少的 SKU
    df["priority_score"] = df["forecast_demand_3d"] / (df["stock_on_hand"] + 1)
    df.sort_values("priority_score", ascending=False, inplace=True)

    remaining_budget = daily_budget
    new_qty = []

    for _, row in df.iterrows():
        price = row["y_price_1d_pred"]
        if price <= 0:
            new_qty.append(row["purchase_qty_rounded"])
            continue

        max_affordable = remaining_budget / price
        original_qty = row["purchase_qty_rounded"]

        if max_affordable >= original_qty:
            # 该 SKU 可以按原建议量采购
            new_qty.append(original_qty)
            remaining_budget -= original_qty * price
        else:
            # 只能部分采购
            adj_qty = max_affordable
            new_qty.append(adj_qty)
            remaining_budget = 0

    df["purchase_qty_rounded"] = [math.floor(q) if q > 0 else 0 for q in new_qty]

    return df


# ======================= 主流程 =======================

def run_procurement():
    """F 步主入口：生成采购建议 CSV。"""
    # 1. 读取成交量预测 & 价格预测
    df_vol = load_volume_forecast()
    df_price = load_price_forecast()

    # run_date / target_date
    df_vol["run_date"] = pd.to_datetime(df_vol["ts"]).dt.date
    df_vol["target_date"] = (
        pd.to_datetime(df_vol["ts"]) + pd.to_timedelta(1, unit="D")
    ).dt.date

    latest_run_date = df_vol["run_date"].max()
    df_vol = df_vol[df_vol["run_date"] == latest_run_date].copy()

    print(f"ℹ 使用 run_date = {latest_run_date} 的预测数据生成采购建议，样本数：{len(df_vol):,}")

    # 2. 合并价格预测
    if df_price is not None:
        candidate_keys = [
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
        ]
        merge_keys = [
            c for c in candidate_keys
            if c in df_vol.columns and c in df_price.columns
        ]
        if not merge_keys:
            raise ValueError("无法在 volume 与 price 预测文件之间找到共同主键列用于合并。")

        df = df_vol.merge(
            df_price[merge_keys + ["y_price_1d_pred"]],
            on=merge_keys,
            how="left",
        )
    else:
        df = df_vol.copy()

    # recent_avg_price 暂留空，将来可用历史均价或花价指数填充
    df["recent_avg_price"] = pd.NA

    # 3. 读取 / 构造库存快照
    if INVENTORY_FROM_DB and get_local_conn is not None:
        try:
            df_inv = load_inventory_from_db(latest_run_date)
            if df_inv.empty:
                print("⚠ 数据库库存快照为空，将退回使用 0 库存。")
                df_inv = make_zero_inventory(df, latest_run_date)
        except Exception as e:
            print(f"⚠ 从数据库读取库存失败：{e}，将退回使用 0 库存。")
            df_inv = make_zero_inventory(df, latest_run_date)
    else:
        df_inv = make_zero_inventory(df, latest_run_date)

    inv_key_cols = [
        "run_date",
        "product_id",
        "variety",
        "grade",
        "market_name",
        "classify_name",
        "spec",
        "color",
        "place",
        "shop_name",
    ]

    df = df.merge(
        df_inv[inv_key_cols + ["stock_on_hand", "stock_in_transit"]],
        on=inv_key_cols,
        how="left",
    )
    df["stock_on_hand"] = df["stock_on_hand"].fillna(0)
    df["stock_in_transit"] = df["stock_in_transit"].fillna(0)

    # 4. 逐行计算采购建议
    df_result = df.apply(compute_procurement_for_row, axis=1)

    # 5. 预算约束（可选）
    df_result = apply_budget_constraint(df_result, DAILY_BUDGET)

    # 6. 输出结果
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ 采购建议已生成：{OUTPUT_PATH}，共 {len(df_result):,} 行")


if __name__ == "__main__":
    run_procurement()
