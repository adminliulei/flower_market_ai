"""
C1：初步清洗（preliminary_cleaning）

- 功能：对从 PostgreSQL 样本表 fm_market_price 读出的原始数据做“弱清洗”：
  * 删除对预测无意义的幽灵字段
  * 统一单位（扎 → 支）
  * 处理明显错误值（价格 ≤ 0、volume < 0）
  * 文本空字符串标准化为缺失值 NaN
"""

from .core import clean_preliminary

__all__ = ["clean_preliminary"]
