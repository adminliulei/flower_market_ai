# src/basic_analysis/run.py
"""
E 步：花价指数计算入口

用法（在项目根目录）：
    python -m src.basic_analysis.run
"""

from __future__ import annotations

from .price_index_core import (
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_CSV,
    run_price_index_pipeline,
)


def main():
    print("🌼 开始执行 E 步：花价指数计算 ...")
    summary = run_price_index_pipeline(
        input_csv=DEFAULT_INPUT_CSV,
        output_csv=DEFAULT_OUTPUT_CSV,
    )

    print("\n📌 花价指数计算摘要：")
    print(f"- 输出记录数：{summary.total_rows:,}")
    print(f"- 覆盖层级：{', '.join(summary.scopes)}")
    print(f"- 覆盖日期范围：{summary.date_range}")
    print("\nE 步完成，可用于后续特征工程。")


if __name__ == "__main__":
    main()
