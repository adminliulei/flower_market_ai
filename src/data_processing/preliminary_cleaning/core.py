import pandas as pd
import numpy as np
from datetime import datetime


def clean_preliminary(df: pd.DataFrame) -> pd.DataFrame:
    """
    C1 初步清洗（弱清洗）
    目标：修复硬错误，不做复杂判断、不做机器学习
    """

    df = df.copy()

    # -------------------------
    # 1. 价格值 < 0 或 = 0 → 设为 NaN
    # -------------------------
    for col in ["wholesale_price", "retail_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] <= 0, col] = np.nan

    # -------------------------
    # 2. 成交量 volume < 0 → 设为 NaN
    # -------------------------
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df.loc[df["volume"] < 0, "volume"] = np.nan

    # -------------------------
    # 3. 文本字段空字符串 → NaN
    # -------------------------
    text_columns = [
        "variety", "grade", "market_name", "classify_name",
        "spec", "color", "place", "shop_name", "unit"
    ]
    for col in text_columns:
        df[col] = df[col].replace("", np.nan)

    # -------------------------
    # 4. 显然错误的批发价/零售价反转纠正
    #    如果 批发价 > 零售价 × 3（明显异常），交换两者
    # -------------------------
    mask = (
        df["wholesale_price"].notna() &
        df["retail_price"].notna() &
        (df["wholesale_price"] > df["retail_price"] * 3)
    )
    df.loc[mask, ["wholesale_price", "retail_price"]] = df.loc[
        mask, ["retail_price", "wholesale_price"]
    ].values

    # -------------------------
    # 5. ingest_at 为 NULL → 补当前时间
    # -------------------------
    df["ingest_at"] = pd.to_datetime(df["ingest_at"], errors="coerce")
    df["ingest_at"] = df["ingest_at"].fillna(datetime.utcnow())

    return df
