# src/prediction_models/short_term_price_pred/run.py
"""
短期价格预测 A 步一键流水线

步骤：
1. 检查 / 生成特征工程结果（time_series_features.csv）
2. 训练 1/2/3 日价格预测模型
3. 在验证集上生成预测结果 CSV
4. 生成评估 PDF 报告

使用方式（在项目根目录）：
    python -m src.prediction_models.short_term_price_pred.run
"""

from __future__ import annotations

from pathlib import Path

# 绝对导入，确保以 `python -m src....` 方式运行时正常
from src.prediction_models.common.feature_engineering import (
    build_features,
    TIME_SERIES_FEATURES_CSV,
)
from src.prediction_models.short_term_price_pred.model_train import main as train_main
from src.prediction_models.short_term_price_pred.model_predict import (
    main as predict_main,
)
from src.prediction_models.common.model_evaluation import generate_report


def main():
    print("🌼 [A 步] 短期价格预测一键流水线启动 ...")

    # 1. 特征工程（若不存在则自动生成）
    if not TIME_SERIES_FEATURES_CSV.exists():
        print(f"🔧 未检测到特征文件：{TIME_SERIES_FEATURES_CSV}")
        print("   -> 自动执行特征工程（Feature Engineering） ...")
        summary = build_features()
        print(
            f"✅ 特征工程完成：样本行数={summary.n_rows:,}，"
            f"特征数={summary.n_features}，日期范围={summary.date_range}"
        )
    else:
        print(f"✅ 检测到特征文件：{TIME_SERIES_FEATURES_CSV}")

    # 2. 模型训练
    print("\n🚀 Step 1 - 训练短期价格预测模型（1/2/3 日）")
    train_main()

    # 3. 生成预测结果 CSV
    print("\n🚀 Step 2 - 在验证集上生成预测结果 CSV")
    predict_main()

    # 4. 生成评估 PDF 报告
    print("\n🚀 Step 3 - 生成价格预测评估 PDF 报告")
    metrics_summary = generate_report()

    print("\n🌈 A 步短期价格预测流水线全部完成。")
    print("   - 预测结果：data/output/price_prediction_result.csv")
    print("   - 评估报告：reports/price_model_eval_report.pdf")


if __name__ == "__main__":
    main()
