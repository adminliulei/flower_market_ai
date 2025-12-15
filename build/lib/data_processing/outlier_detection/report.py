# src/data_processing/outlier_detection/report.py
"""
C2 强清洗：统计 + PDF 报告模块（支持中文字体，不乱码）

职责：
1. 对带异常标签的数据进行统计；
2. 生成异常检测质量报告 PDF；
3. 提供 run_full_outlier_pipeline() 一站式执行：
    - 读取 D 步输出
    - 调用 core.detect_outliers
    - 保存 cleaned 数据
    - 生成 PDF 报告
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from .core import (
    ROOT,
    TS_COL,
    PRICE_COL,
    VOLUME_COL,
    GROUP_KEYS,
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_CSV,
    detect_outliers,
)

# -------------------------
# 字体设置（防止 PDF 中文乱码）
# -------------------------
# 常见可用中文字体：SimHei（黑体）、Microsoft YaHei（微软雅黑）
# 如果其中一个字体可用，matplotlib 将使用它渲染中文
for font in ["Microsoft YaHei", "SimHei", "STSong"]:
    try:
        matplotlib.font_manager.findfont(font, fallback_to_default=False)
        matplotlib.rcParams["font.family"] = font
        break
    except Exception:
        continue

matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题

# -------------------------
# 路径
# -------------------------

REPORTS_DIR = ROOT / "reports"
DEFAULT_REPORT_PDF = REPORTS_DIR / "c2_outlier_quality_report.pdf"


# -------------------------
# 数据统计结果结构体
# -------------------------

@dataclass
class OutlierStats:
    total_rows: int
    price_outliers: int
    volume_outliers: int
    price_outlier_ratio: float
    volume_outlier_ratio: float

    daily_price_outliers: pd.DataFrame
    daily_volume_outliers: pd.DataFrame
    variety_price_outliers: pd.DataFrame
    variety_volume_outliers: pd.DataFrame


# -------------------------
# 统计函数
# -------------------------

def compute_outlier_stats(df: pd.DataFrame) -> OutlierStats:
    """根据 df 中异常标签字段生成统计摘要。"""
    total_rows = len(df)
    price_outliers = int(df["is_outlier_price"].sum())
    volume_outliers = int(df["is_outlier_volume"].sum())

    price_ratio = price_outliers / total_rows if total_rows else 0
    volume_ratio = volume_outliers / total_rows if total_rows else 0

    # 每日异常情况
    daily_price = (
        df.groupby(TS_COL)["is_outlier_price"]
        .agg(count="sum", total="count")
        .assign(ratio=lambda x: x["count"] / x["total"])
    )[["count", "ratio"]]

    daily_volume = (
        df.groupby(TS_COL)["is_outlier_volume"]
        .agg(count="sum", total="count")
        .assign(ratio=lambda x: x["count"] / x["total"])
    )[["count", "ratio"]]

    # 品种维度 Top20
    if "variety" in df.columns:
        variety_price = (
            df.groupby("variety")["is_outlier_price"]
            .agg(count="sum", total="count")
            .assign(ratio=lambda x: x["count"] / x["total"])
            .reset_index()
            .sort_values("ratio", ascending=False)
            .head(20)
        )

        variety_volume = (
            df.groupby("variety")["is_outlier_volume"]
            .agg(count="sum", total="count")
            .assign(ratio=lambda x: x["count"] / x["total"])
            .reset_index()
            .sort_values("ratio", ascending=False)
            .head(20)
        )
    else:
        variety_price = pd.DataFrame(columns=["variety", "count", "ratio"])
        variety_volume = pd.DataFrame(columns=["variety", "count", "ratio"])

    return OutlierStats(
        total_rows=total_rows,
        price_outliers=price_outliers,
        volume_outliers=volume_outliers,
        price_outlier_ratio=price_ratio,
        volume_outlier_ratio=volume_ratio,
        daily_price_outliers=daily_price,
        daily_volume_outliers=daily_volume,
        variety_price_outliers=variety_price,
        variety_volume_outliers=variety_volume,
    )


# -------------------------
# PDF 报告生成
# -------------------------

def generate_outlier_report_pdf(
    df: pd.DataFrame,
    stats: OutlierStats,
    output_path: Path = DEFAULT_REPORT_PDF,
    max_varieties: int = 10,
):
    """生成可视化 PDF 报告（中文字体保证不乱码）"""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        # ----- Page 1：总体概览 -----
        fig1 = plt.figure(figsize=(8.27, 11.69))  # A4 尺寸
        plt.suptitle("C2 异常检测质量报告 - 概览", fontsize=18)

        text = [
            f"总记录数：{stats.total_rows:,}",
            f"价格异常数量：{stats.price_outliers:,} ({stats.price_outlier_ratio:.2%})",
            f"成交量异常数量：{stats.volume_outliers:,} ({stats.volume_outlier_ratio:.2%})",
            "",
            "说明：",
            "1. 异常标签由多维规则生成：Z 分数、跳变比例、涨跌幅、波动率、季节性偏移等；",
            "2. 异常点未被删除，而是使用 is_outlier_price / is_outlier_volume 打标签；",
            "3. 标注后的数据用于模型训练，提高鲁棒性与抗噪能力；",
        ]

        plt.axis("off")
        plt.text(0.05, 0.95, "\n".join(text), fontsize=12, va="top")
        pdf.savefig(fig1)
        plt.close(fig1)

        # ----- Page 2：每日价格异常数量 -----
        if not stats.daily_price_outliers.empty:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            daily = stats.daily_price_outliers.sort_index()
            ax2.plot(daily.index, daily["count"], marker="o", linewidth=1)
            ax2.set_title("每日价格异常数量")
            ax2.set_xlabel("日期")
            ax2.set_ylabel("异常数量")
            fig2.autofmt_xdate()
            pdf.savefig(fig2)
            plt.close(fig2)

        # ----- Page 3：按品种的价格异常比例 -----
        if not stats.variety_price_outliers.empty:
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            top = stats.variety_price_outliers.head(max_varieties)
            ax3.bar(top["variety"], top["ratio"])
            ax3.set_title("按品种统计的价格异常比例（Top N）")
            ax3.set_ylabel("异常比例")
            ax3.set_xticklabels(top["variety"], rotation=40, ha="right")
            pdf.savefig(fig3)
            plt.close(fig3)

        # ----- Page 4：典型时间序列案例 -----
        if df["is_outlier_price"].any():
            grp = (
                df.groupby(GROUP_KEYS)["is_outlier_price"]
                .sum()
                .sort_values(ascending=False)
            )
            top_key = grp.index[0]

            mask = pd.Series(True, index=df.index)
            for col, val in zip(GROUP_KEYS, top_key):
                mask &= df[col] == val

            example = df[mask].sort_values(TS_COL)

            fig4, ax4 = plt.subplots(figsize=(10, 4))
            ax4.plot(example[TS_COL], example[PRICE_COL], label="价格变化")
            abnormal = example[example["is_outlier_price"]]
            ax4.scatter(
                abnormal[TS_COL],
                abnormal[PRICE_COL],
                color="red",
                s=35,
                label="异常点",
            )

            title_suffix = ", ".join(f"{c}={v}" for c, v in zip(GROUP_KEYS, top_key))
            ax4.set_title(f"典型时间序列案例（价格异常）\n{title_suffix}")
            ax4.legend()
            fig4.autofmt_xdate()
            pdf.savefig(fig4)
            plt.close(fig4)


# -------------------------
# 一站式入口（供 run.py 调用）
# -------------------------

def run_full_outlier_pipeline(
    input_csv: Path = DEFAULT_INPUT_CSV,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    report_pdf: Path = DEFAULT_REPORT_PDF,
    use_isolation_forest: bool = False,
) -> OutlierStats:
    """完整执行 C2 并生成 PDF 报告"""

    print(f"📥 正在加载 D 步结果：{input_csv}")
    df = pd.read_csv(input_csv)

    print("🧹 执行 C2 强清洗：异常检测 ...")
    df_out = detect_outliers(df, use_isolation_forest=use_isolation_forest)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"✅ 已输出 cleaned 数据：{output_csv}")

    print("📊 正在计算统计指标并生成 PDF 报告 ...")
    stats = compute_outlier_stats(df_out)
    generate_outlier_report_pdf(df_out, stats, output_path=report_pdf)
    print(f"✅ PDF 报告已生成：{report_pdf}")

    return stats
