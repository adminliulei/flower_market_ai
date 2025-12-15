# src/data_processing/outlier_detection/run.py
"""
C2 强清洗：命令行入口

用法：
    python -m src.data_processing.outlier_detection.run
"""

from __future__ import annotations

from .core import ROOT, DEFAULT_INPUT_CSV, DEFAULT_OUTPUT_CSV
from .report import DEFAULT_REPORT_PDF, run_full_outlier_pipeline


def main():
    input_csv = DEFAULT_INPUT_CSV
    output_csv = DEFAULT_OUTPUT_CSV
    report_pdf = DEFAULT_REPORT_PDF

    print("🌼 开始执行 C2 强清洗（异常检测） ...")
    print(f"项目根目录：{ROOT}")

    stats = run_full_outlier_pipeline(
        input_csv=input_csv,
        output_csv=output_csv,
        report_pdf=report_pdf,
        use_isolation_forest=False,  # 如需启用 IsolationForest，可改为 True
    )

    print("\n📌 C2 清洗摘要：")
    print(f"- 总记录数：{stats.total_rows:,}")
    print(
        f"- 价格异常：{stats.price_outliers:,} "
        f"({stats.price_outlier_ratio:.2%})"
    )
    print(
        f"- 成交量异常：{stats.volume_outliers:,} "
        f"({stats.volume_outlier_ratio:.2%})"
    )
    print("\nC2 强清洗完成。")


if __name__ == "__main__":
    main()
