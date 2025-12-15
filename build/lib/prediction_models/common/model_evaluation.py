# src/prediction_models/common/model_evaluation.py
"""
价格预测结果评估脚本（生成 PDF 报告）

输入：
    data/output/price_prediction_result.csv
        由 short_term_price_pred.model_predict 生成

输出：
    reports/price_model_eval_report.pdf

运行方式（项目根目录）：
    python -m src.prediction_models.common.model_evaluation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_CSV = OUTPUT_DIR / "price_prediction_result.csv"
REPORT_PDF = REPORT_DIR / "price_model_eval_report.pdf"

TS_COL = "ts"


@dataclass
class HorizonMetrics:
    horizon: int
    n_samples: int
    mae: float
    rmse: float
    mape: float
    median_ape: float
    p90_ape: float
    p95_ape: float


def _setup_matplotlib():
    plt.rcParams["axes.unicode_minus"] = False
    for font in ["SimHei", "Microsoft YaHei", "STHeiti"]:
        plt.rcParams["font.sans-serif"] = [font]
        break


def load_result_data() -> pd.DataFrame:
    if not RESULT_CSV.exists():
        raise FileNotFoundError(f"未找到预测结果文件：{RESULT_CSV}")
    df = pd.read_csv(RESULT_CSV)
    df[TS_COL] = pd.to_datetime(df[TS_COL])
    return df


def compute_horizon_metrics(df: pd.DataFrame) -> List[HorizonMetrics]:
    metrics: List[HorizonMetrics] = []
    for h, g in df.groupby("horizon"):
        n = len(g)
        mae = float(g["abs_error"].mean())
        rmse = float(np.sqrt((g["error"] ** 2).mean()))
        mape = float(g["ape"].mean())
        median_ape = float(g["ape"].median())
        p90_ape = float(np.percentile(g["ape"], 90))
        p95_ape = float(np.percentile(g["ape"], 95))

        metrics.append(
            HorizonMetrics(
                horizon=int(h),
                n_samples=n,
                mae=mae,
                rmse=rmse,
                mape=mape,
                median_ape=median_ape,
                p90_ape=p90_ape,
                p95_ape=p95_ape,
            )
        )

    metrics.sort(key=lambda x: x.horizon)
    return metrics


# ---------- 画图 ----------

def add_title_page(pdf: PdfPages, metrics: List[HorizonMetrics]):
    fig = plt.figure(figsize=(10, 6))
    plt.axis("off")

    title = "鲜花市场短期价格预测模型评估报告"
    subtitle = "Price Model Evaluation Report"

    fig.text(0.5, 0.8, title, ha="center", va="center", fontsize=20, weight="bold")
    fig.text(0.5, 0.74, subtitle, ha="center", va="center", fontsize=11)

    y = 0.64
    total_n = sum(m.n_samples for m in metrics)
    fig.text(0.08, y, f"样本总数：{total_n:,}", fontsize=11)
    y -= 0.04
    fig.text(0.08, y, "各预测期整体表现：", fontsize=11)
    y -= 0.04

    for m in metrics:
        line = (
            f"· {m.horizon} 日预测："
            f"MAE={m.mae:.4f}，RMSE={m.rmse:.4f}，"
            f"MAPE={m.mape:.2f}%（中位数APE={m.median_ape:.2f}%，"
            f"P90={m.p90_ape:.2f}% / P95={m.p95_ape:.2f}%）"
        )
        fig.text(0.10, y, line, fontsize=10)
        y -= 0.035

    fig.text(
        0.08,
        0.12,
        "说明：本报告基于验证集预测结果，评估 1/2/3 日价格预测模型效果，用于业务汇报与模型迭代优化。",
        fontsize=9,
        color="gray",
    )

    pdf.savefig(fig)
    plt.close(fig)


def plot_ape_histograms(pdf: PdfPages, df: pd.DataFrame, metrics: List[HorizonMetrics]):
    for m in metrics:
        g = df[df["horizon"] == m.horizon]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(g["ape"], bins=60, alpha=0.8)
        ax.set_title(f"{m.horizon} 日预测：绝对百分比误差分布（APE）")
        ax.set_xlabel("APE (%)")
        ax.set_ylabel("样本数量")

        txt = (
            f"样本数={m.n_samples:,}\n"
            f"平均APE={m.mape:.2f}% 中位数={m.median_ape:.2f}%\n"
            f"P90={m.p90_ape:.2f}%  P95={m.p95_ape:.2f}%"
        )
        ax.text(
            0.98,
            0.98,
            txt,
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        pdf.savefig(fig)
        plt.close(fig)


def plot_true_vs_pred_scatter(pdf: PdfPages, df: pd.DataFrame, metrics: List[HorizonMetrics]):
    for m in metrics:
        g = df[df["horizon"] == m.horizon].copy()

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(g["y_true"], g["y_pred"], s=4, alpha=0.25)

        min_v = float(min(g["y_true"].min(), g["y_pred"].min()))
        max_v = float(max(g["y_true"].max(), g["y_pred"].max()))
        ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)

        ax.set_title(f"{m.horizon} 日预测：真实值 vs 预测值散点图")
        ax.set_xlabel("真实价格 y_true")
        ax.set_ylabel("预测价格 y_pred")

        pdf.savefig(fig)
        plt.close(fig)


def plot_ape_by_price_bucket(pdf: PdfPages, df: pd.DataFrame):
    """
    按真实价格分桶，看不同价位段的平均 APE。
    例如：0-2, 2-4, 4-6, 6-8, 8-10, 10+（元）
    """
    price = df["y_true"].values
    bins = [0, 2, 4, 6, 8, 10, np.inf]
    labels = ["0-2", "2-4", "4-6", "6-8", "8-10", "10+"]

    df_bucket = df.copy()
    df_bucket["price_bucket"] = pd.cut(price, bins=bins, labels=labels, right=False)

    grouped = (
        df_bucket.groupby(["horizon", "price_bucket"], observed=False)["ape"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for h in sorted(df_bucket["horizon"].unique()):
        sub = grouped[grouped["horizon"] == h]
        ax.plot(sub["price_bucket"].astype(str), sub["mean"], marker="o", label=f"{h} 日")

    ax.set_title("不同真实价格区间的平均 APE（按 horizon）")
    ax.set_xlabel("真实价格区间（元）")
    ax.set_ylabel("平均 APE (%)")
    ax.legend(title="预测期", fontsize=9)

    pdf.savefig(fig)
    plt.close(fig)


def plot_time_series_examples(pdf: PdfPages, df: pd.DataFrame):
    """
    挑选几个品种的 1 日预测，画时间序列对比
    """
    df1 = df[df["horizon"] == 1].copy()

    if "variety" in df1.columns:
        varieties = df1["variety"].value_counts().head(4).index.tolist()
    else:
        varieties = [None]

    for v in varieties:
        if v is None:
            g = df1.sort_values(TS_COL).tail(200)
            title_prefix = "示例（全部品种混合）"
        else:
            g = df1[df1["variety"] == v].sort_values(TS_COL).tail(200)
            title_prefix = f"品种：{v}"

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(g[TS_COL], g["y_true"], label="真实价格 y_true", linewidth=1.0)
        ax.plot(g[TS_COL], g["y_pred"], label="预测价格 y_pred", linewidth=1.0, linestyle="--")
        ax.set_title(f"{title_prefix} - 1 日预测时间序列对比（最近 200 点）")
        ax.set_xlabel("日期")
        ax.set_ylabel("价格")
        ax.legend(fontsize=9)
        fig.autofmt_xdate()

        pdf.savefig(fig)
        plt.close(fig)


# ---------- 主流程 ----------

def generate_report() -> Dict[str, float]:
    _setup_matplotlib()

    print(f"📥 读取预测结果：{RESULT_CSV}")
    df = load_result_data()

    required_cols = {"horizon", "y_true", "y_pred"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"结果 CSV 缺少必要字段：{required_cols - set(df.columns)}")

    if "error" not in df.columns or "abs_error" not in df.columns or "ape" not in df.columns:
        eps = 1e-6
        df["error"] = df["y_pred"] - df["y_true"]
        df["abs_error"] = df["error"].abs()
        df["ape"] = df["abs_error"] / (np.abs(df["y_true"]) + eps) * 100.0

    metrics = compute_horizon_metrics(df)

    print("🧮 各 horizon 指标：")
    for m in metrics:
        print(
            f"- {m.horizon} 日：N={m.n_samples:,}，"
            f"MAE={m.mae:.4f}，RMSE={m.rmse:.4f}，"
            f"MAPE={m.mape:.2f}%，中位APE={m.median_ape:.2f}%，"
            f"P90={m.p90_ape:.2f}% / P95={m.p95_ape:.2f}%"
        )

    print(f"\n📝 生成 PDF 报告：{REPORT_PDF}")
    with PdfPages(REPORT_PDF) as pdf:
        add_title_page(pdf, metrics)
        plot_ape_histograms(pdf, df, metrics)
        plot_true_vs_pred_scatter(pdf, df, metrics)
        plot_ape_by_price_bucket(pdf, df)
        plot_time_series_examples(pdf, df)

    print("✅ 报告生成完成。")
    return {f"mae_{m.horizon}d": m.mae for m in metrics}


def main():
    generate_report()


if __name__ == "__main__":
    main()
