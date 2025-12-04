# 在 src/tests/test_data_processing.py 中追加以下内容

import pandas as pd
from datetime import datetime, timedelta


def test_outlier_detection_basic():
    """
    C2 异常检测的基础行为测试：
    构造一条平稳序列 + 一个明显异常点，看是否能被标记。
    """
    from src.data_processing.outlier_detection.core import detect_outliers

    # 构造 15 天数据，其中第 10 天价格暴涨 10 倍、成交量暴涨 20 倍
    base_date = datetime(2025, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(15)]

    data = []
    for i, ts in enumerate(dates):
        price = 10.0
        volume = 100.0
        if i == 10:
            price = 100.0
            volume = 2000.0
        data.append(
            {
                "ts": ts,
                "product_id": 1,
                "variety": "测试玫瑰",
                "spec": "10枝/扎",
                "grade": "A",
                "shop_name": "测试商户",
                "classify_name": "玫瑰",
                "color": "红",
                "retail_price": price,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)

    df_out = detect_outliers(df, use_isolation_forest=False)

    # 检查字段是否存在
    assert "is_outlier_price" in df_out.columns
    assert "is_outlier_volume" in df_out.columns

    # 第 10 行应为异常
    row10 = df_out.iloc[10]
    assert bool(row10["is_outlier_price"]) is True
    assert bool(row10["is_outlier_volume"]) is True

    # 大部分其它行不应被标记为异常
    normal_flags = df_out["is_outlier_price"].sum()
    assert normal_flags >= 1  # 至少有一个（第 10 行）
    assert normal_flags <= 3  # 不应大量误判
