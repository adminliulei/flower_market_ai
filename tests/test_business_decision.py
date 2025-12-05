# tests/test_business_decision.py
from pathlib import Path
import pandas as pd


# 项目根目录：.../flower_market_ai
ROOT = Path(__file__).resolve().parents[1]

DATA_OUTPUT = ROOT / "data" / "output"
PRICE_PRED_WIDE = DATA_OUTPUT / "price_prediction_wide.csv"


def test_can_load_price_prediction_wide():
    """只是验证一下文件路径和格式是否正常。"""
    assert PRICE_PRED_WIDE.exists(), f"文件不存在：{PRICE_PRED_WIDE}"

    df = pd.read_csv(PRICE_PRED_WIDE)
    # 这里随便看几列，后面你可以继续加业务规则测试
    print(df.head())
    assert {"ts", "product_id", "variety",
            "pred_price_1d", "pred_price_2d", "pred_price_3d"} <= set(df.columns)
