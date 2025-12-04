import re
from typing import List

import numpy as np
import pandas as pd

# 在 C1 阶段就要彻底移除的“幽灵字段”
GHOST_COLS: List[str] = [
    "wholesale_price",  # 实际生产中未写入，长期为缺失
    "image_url",        # 图片地址，大字段，不参与建模
    "images",           # 图片 JSON/列表，同上
    "ingest_at",        # 入库时间戳，对价格预测无直接价值
]


def _drop_ghost_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    在 C1 初步清洗阶段移除对预测无意义、且经常为空或体积巨大的字段。
    这样后续 D / C2 / 特征工程 / 建模都不再受这些“幽灵特征”干扰。
    """
    cols_to_drop = [c for c in GHOST_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(
            "🧹 C1：已移除幽灵字段："
            + ", ".join(cols_to_drop)
            + "（后续流程与建模中不再使用）。"
        )
    return df


def _normalize_unit_and_spec(df: pd.DataFrame) -> pd.DataFrame:
    """
    将不同单位统一转换成“支”作为内部标准单位，并做基础数值清洗。

    场景示例：
        unit = '支', spec 为空              → 直接视为按支计价
        unit = '扎', spec = '10枝/扎'      → 1 扎 = 10 支
        unit = '扎', spec = '20枝/扎'      → 1 扎 = 20 支

    处理逻辑：
        - 对 unit='扎' 的记录：
            retail_price := 单支价格 = 原价 / 枝数
            volume      := 支数     = 原成交量 * 枝数
            unit        := '支'
        - 基础清洗：
            retail_price <= 0 → NaN
            volume < 0 → NaN
    """
    if "retail_price" not in df.columns or "volume" not in df.columns:
        return df

    # 先把价格、成交量转为数值并做基础清洗
    df["retail_price"] = pd.to_numeric(df["retail_price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df.loc[df["retail_price"] <= 0, "retail_price"] = np.nan
    df.loc[df["volume"] < 0, "volume"] = np.nan

    # 如果缺少 unit/spec，就到此为止
    if "unit" not in df.columns or "spec" not in df.columns:
        # volume 用可空整型表示
        df["volume"] = df["volume"].round().astype("Int64")
        return df

    df = df.copy()

    def parse_stem_count(spec: object, unit_val: object) -> int:
        """
        从 spec 中解析出“每扎的枝数”：
            '10枝/扎'、'10支/扎'、' 20枝 / 扎 ' → 10 / 20
        若无法解析：
            - 如果 unit 是 '扎'，保守默认 10
            - 否则返回 1（按支计）
        """
        if isinstance(spec, str):
            m = re.search(r"(\d+)\s*[枝支]", spec)
            if m:
                return int(m.group(1))

        if isinstance(unit_val, str) and unit_val.strip() == "扎":
            return 10

        return 1

    # 规范一下 unit 字段
    df["unit"] = df["unit"].astype(str).str.strip()

    # 计算每条记录对应的枝数（对 unit='扎' 特别重要）
    df["stem_count"] = [
        parse_stem_count(spec, unit_val)
        for spec, unit_val in zip(df["spec"], df["unit"])
    ]

    # 只对 unit = '扎' 且 stem_count > 0 的记录做换算
    mask_bundle = df["unit"] == "扎"
    mask_valid = mask_bundle & (df["stem_count"] > 0)

    # 零售价：从“每扎价格”换算成“每支价格”
    df.loc[mask_valid, "retail_price"] = (
        df.loc[mask_valid, "retail_price"] / df.loc[mask_valid, "stem_count"]
    )

    # 成交量：从“扎数”换算成“支数”
    df.loc[mask_valid, "volume"] = (
        df.loc[mask_valid, "volume"] * df.loc[mask_valid, "stem_count"]
    )

    # volume 用可空整型表示
    df["volume"] = df["volume"].round().astype("Int64")

    # 最终将所有“扎”统一视为“支”
    df.loc[mask_bundle, "unit"] = "支"

    # stem_count 是中间计算字段，用完即可删除
    df = df.drop(columns=["stem_count"])

    return df


def clean_preliminary(df: pd.DataFrame) -> pd.DataFrame:
    """
    C1 初步清洗（弱清洗）
    目标：修复硬错误，不做复杂判断、不做机器学习

    处理内容：
    1. 删除幽灵字段（wholesale_price / image_url / images / ingest_at）
    2. 基础数值清洗 + 单位换算：
       - retail_price <= 0 → NaN
       - volume < 0 → NaN
       - unit='扎' 且 spec=10枝/扎 等 → 换算成“支价 + 支数”，unit 统一为“支”
    3. 文本字段中空字符串 → NaN
    4. 零售价最终统一保留两位小数
    """

    df = df.copy()

    # 0. 删除不会参与预测的幽灵字段
    df = _drop_ghost_columns(df)

    # 1 & 2. 数值清洗 + 单位统一
    df = _normalize_unit_and_spec(df)

    # 3. 文本字段空字符串 → NaN
    text_columns = [
        "variety",
        "grade",
        "market_name",
        "classify_name",
        "spec",
        "color",
        "place",
        "shop_name",
        "unit",
    ]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].replace("", np.nan)

    # 4. 最终保证零售价保留两位小数
    if "retail_price" in df.columns:
        df["retail_price"] = pd.to_numeric(df["retail_price"], errors="coerce").round(2)

    return df
