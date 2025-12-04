"""
D 步：整天缺失补全（missing_value_filling）

核心逻辑：
- 按花 + 规格 + 档口等维度拆成时间序列
- 对每个序列，只在“生命区间” [t_min, t_max] 内补齐中间缺日
- 补全时只填“被前后真实数据夹住的缺口”，用时间插值实现前后平滑过渡
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

# 一个时间序列的粒度（可以根据实际情况调整顺序/字段）
# 注意：下面这些字段并不一定全部存在，代码会自动过滤掉不存在的字段
GROUP_KEY_CANDIDATES: List[str] = [
    "product_id",
    "variety",
    "spec",
    "grade",
    "shop_name",
    "classify_name",
    "color",
]

# 需要补值的数值列
NUM_COLS: List[str] = ["retail_price", "volume"]


def _normalize_ts(df: pd.DataFrame) -> pd.DataFrame:
    """将 ts 统一为去掉时间部分的日期（datetime64[ns]，时间为 00:00:00）"""
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"]).dt.normalize()
    return out


def _detect_group_cols(df: pd.DataFrame) -> List[str]:
    """从候选字段中选出数据里实际存在的分组字段。"""
    group_cols = [c for c in GROUP_KEY_CANDIDATES if c in df.columns]
    if not group_cols:
        raise ValueError(
            f"D 步补全失败：在当前数据中找不到任何分组字段，"
            f"请至少保证 {GROUP_KEY_CANDIDATES} 中有一列存在。"
        )
    return group_cols


def _detect_static_cols(df: pd.DataFrame) -> List[str]:
    """
    自动推断需要当作“静态属性”的列：
    - 排除 ts、本身要插值的数值列、id 等
    - 其余所有列都视为静态列，对新增日期用 ffill+bfill 补齐
    """
    exclude = {"ts", "id"}
    exclude.update(NUM_COLS)

    static_cols = [c for c in df.columns if c not in exclude]
    return static_cols


def _fill_one_group(g: pd.DataFrame, static_cols: List[str]) -> pd.DataFrame:
    """
    对单个 GROUP（例如同一个 product_id + variety + spec 等组合）做整天补全。

    规则：
    - 生命区间：[t_min, t_max] = 该组最早和最晚真实记录的日期
    - 只在生命区间内按日重建时间轴并补齐缺口
    - 对被前后真实点夹住的缺口，用时间插值进行平滑补值
    - 对序列两端（只有一侧有真实点）的日期不做补
    """
    if g.empty:
        return g

    # 按日期排序
    g = g.sort_values("ts").copy()
    g["ts"] = pd.to_datetime(g["ts"]).dt.normalize()

    # 该组真实记录的日期集合
    real_dates = pd.DatetimeIndex(g["ts"].unique())
    t_min = real_dates.min()
    t_max = real_dates.max()

    # 如果这个序列只有 1 天，根本没有“整天缺失”的空间，直接返回并补标记字段
    if t_min == t_max:
        g["is_synthetic_row"] = False
        g["is_filled_retail_price"] = False
        g["is_filled_volume"] = False
        return g

    # 序列生命区间内的完整日期轴
    full_index = pd.date_range(t_min, t_max, freq="D", name="ts")

    # 先把原始索引保存下来，用来判断哪些是新增日期
    original_index = pd.DatetimeIndex(g["ts"])
    original_date_set = set(original_index)

    # 以 ts 为索引重建时间轴（只在生命区间内）
    g = g.set_index("ts")

    # 先 reindex，得到“有洞”的时间序列
    g_full = g.reindex(full_index)

    # 静态字段：ffill + bfill，保证新增日期也有合理的静态属性
    for col in static_cols:
        if col in g_full.columns:
            g_full[col] = (
                g_full[col]
                .ffill()
                .bfill()
                .infer_objects(copy=False)  # 防止未来 pandas 下架隐式 downcasting
            )

    # 标记哪些行是“系统新增日期”
    is_synthetic_row = ~g_full.index.isin(original_date_set)
    g_full["is_synthetic_row"] = is_synthetic_row

    # 数值字段：时间插值（仅在左右都有真实锚点的“内部缺口”插）
    for col in NUM_COLS:
        if col not in g_full.columns:
            # 确保有对应的标记列
            g_full[f"is_filled_{col}"] = False
            continue

        # 转成 float 方便插值
        series_float = pd.to_numeric(g_full[col], errors="coerce")

        # 线性插值（基于时间索引），只对内部缺口插值
        filled = series_float.interpolate(
            method="time",
            limit_area="inside",  # 只在被前后非空值夹住的 NaN 段插值
        )

        # 写回结果
        if col == "volume":
            # volume 一般为整数，用 Int64（可空整型）
            g_full[col] = filled.round().astype("Int64")
        else:
            # 零售价保留两位小数
            g_full[col] = filled.round(2)

        # 只有“新增日期”且根据插值得到了数值，才认为该字段在 D 步被补过
        g_full[f"is_filled_{col}"] = is_synthetic_row & filled.notna()

    # 还原 ts 为普通列
    g_full = g_full.reset_index()  # index 名称就是 "ts"

    return g_full


def fill_missing_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    D 步总入口：整天缺失补全。

    输入：
        - C1 初步清洗后的数据（market_price_prelim_clean.csv 读入的 DataFrame）
    输出：
        - 在各个时间序列的生命区间内补齐中间缺日后的 DataFrame，
          增加了 is_synthetic_row / is_filled_* 等标记字段。
    """
    if df.empty:
        return df.copy()

    df = _normalize_ts(df)
    group_cols = _detect_group_cols(df)
    static_cols = _detect_static_cols(df)

    filled_groups = []
    for _, g in df.groupby(group_cols, dropna=False):
        filled = _fill_one_group(g, static_cols)
        filled_groups.append(filled)

    full_df = pd.concat(filled_groups, ignore_index=True)

    # 最后可以按日期和品类等排序，方便阅读和后续处理
    sort_cols = [
        c
        for c in [
            "ts",
            "market_name",
            "classify_name",
            "product_id",
            "variety",
            "grade",
        ]
        if c in full_df.columns
    ]
    if sort_cols:
        full_df = full_df.sort_values(sort_cols).reset_index(drop=True)

    # 统一保证零售价保留两位小数
    if "retail_price" in full_df.columns:
        full_df["retail_price"] = (
            pd.to_numeric(full_df["retail_price"], errors="coerce").round(2)
        )

    return full_df
