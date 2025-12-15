# src/basic_analysis/price_index_visual_report.py
"""
E 步：花价指数可视化报告（PDF）

功能：
1. 从 data/intermediate/indices/flower_price_index.csv 读取指数数据；
2. 生成多页 PDF 报告：
   - 全市场花价指数走势（含 MA7 / MA30）
   - 按大类的指数对比（Top N）
   - 某个典型大类或品种的时间序列曲线

输出：
    reports/price_index_visual_report.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


# 项目路径（和之前 price_index_core 一致）
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
INDICES_DIR = INTERMEDIATE_DIR / "indices"
REPORTS_DIR = ROOT / "reports"

INDEX_CSV = INDICES_DIR / "flower_price_index.csv"
OUTPUT_PDF = REPORTS_DIR / "price_index_visual_report.pdf"

# 字体设置（中文不乱码）
for font in ["Microsoft YaHei", "SimHei", "STSong"]:
    try:
        matplotlib.font_manager.findfont(font, fallback_to_default=False)
        matplotlib.rcParams["font.family"] = font
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False


def load_index_df() -> pd.DataFrame:
    df = pd.read_csv(INDEX_CSV)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def plot_overall_index(df: pd.DataFrame, pdf: PdfPages):
    """Page 1：全市场花价指数走势"""
    overall = df[df["scope_type"] == "all"].sort_values("ts")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(overall["ts"], overall["price_index"], label="全市场指数")
    if "index_ma7" in overall.columns:
        ax.plot(overall["ts"], overall["index_ma7"], linestyle="--", label="7 日均线")
    if "index_ma30" in overall.columns:
        ax.plot(overall["ts"], overall["index_ma30"], linestyle=":", label="30 日均线")

    ax.set_title("全市场花价指数走势")
    ax.set_xlabel("日期")
    ax.set_ylabel("成交量加权平均价")
    ax.legend()
    fig.autofmt_xdate()
    pdf.savefig(fig)
    plt.close(fig)


def plot_classify_index_topN(df: pd.DataFrame, pdf: PdfPages, top_n: int = 5):
    """Page 2：按大类的指数对比（选取成交量 Top N）"""
    classify_df = df[df["scope_type"] == "classify"].copy()
    if classify_df.empty:
        return

    # 选出总成交量最大的前 N 个大类
    total_vol = (
        classify_df.groupby("scope_value")["total_volume"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    top_names = total_vol.index.tolist()
    sub = classify_df[classify_df["scope_value"].isin(top_names)].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    for name in top_names:
        d = sub[sub["scope_value"] == name].sort_values("ts")
        ax.plot(d["ts"], d["price_index"], label=name)

    ax.set_title(f"按大类的花价指数走势（成交量 Top {top_n}）")
    ax.set_xlabel("日期")
    ax.set_ylabel("指数价")
    ax.legend()
    fig.autofmt_xdate()
    pdf.savefig(fig)
    plt.close(fig)


def plot_single_scope_series(df: pd.DataFrame, pdf: PdfPages):
    """Page 3：选择一个典型的大类或品种，画时间序列曲线"""
    # 优先选 variety，其次 classify
    if (df["scope_type"] == "variety").any():
        scope_type = "variety"
    elif (df["scope_type"] == "classify").any():
        scope_type = "classify"
    else:
        return

    sub = df[df["scope_type"] == scope_type].copy()
    # 选出异常最多或者成交量最大的一个
    # 这里简单用总成交量 Top 1
    total_vol = (
        sub.groupby("scope_value")["total_volume"]
        .sum()
        .sort_values(ascending=False)
    )
    target_name = total_vol.index[0]
    target = sub[sub["scope_value"] == target_name].sort_values("ts")

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(target["ts"], target["price_index"], label="指数价")
    if "index_ma7" in target.columns:
        ax1.plot(target["ts"], target["index_ma7"], linestyle="--", label="7 日均线")
    ax1.set_xlabel("日期")
    ax1.set_ylabel("指数价")
    ax1.set_title(f"{scope_type} = {target_name} 的花价指数时间序列")
    ax1.legend(loc="upper left")
    fig.autofmt_xdate()

    pdf.savefig(fig)
    plt.close(fig)


def generate_price_index_visual_report():
    df = load_index_df()
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(OUTPUT_PDF) as pdf:
        plot_overall_index(df, pdf)
        plot_classify_index_topN(df, pdf, top_n=5)
        plot_single_scope_series(df, pdf)

    print(f"✅ 花价指数可视化报告已生成：{OUTPUT_PDF}")


if __name__ == "__main__":
    generate_price_index_visual_report()
