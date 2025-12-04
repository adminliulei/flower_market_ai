"""
D 步：整天缺失补全 质量评估报告（PDF）

功能：
- 读取 C1 输出和 D 步输出：
    data/processed/market_price_prelim_clean.csv
    data/processed/market_price_filled.csv

- 生成一个 PDF 报告，内容包括：
    1）整体数据量与补全情况总览（文字页）
    2）各时间序列补全占比分布直方图
    3）按日期统计每天新增补全记录数量的折线图（含每日补全占比）
    4）若干代表性时间序列（product_id + market_name + spec + grade + shop_name）
       的零售价走势，其中原始记录为蓝点，补全记录为红点

输出：
    reports/d_filling_quality_report.pdf
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages

# ======================
# 全局配置
# ======================

# 项目根目录：.../flower_market_ai
ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
REPORT_PATH = REPORT_DIR / "d_filling_quality_report.pdf"

PRELIM_PATH = DATA_DIR / "market_price_prelim_clean.csv"
FILLED_PATH = DATA_DIR / "market_price_filled.csv"

# 与 D 步补全时保持一致的分组粒度
GROUP_KEY_CANDIDATES: List[str] = [
    "product_id",
    "variety",
    "spec",
    "grade",
    "shop_name",
    "classify_name",
    "color",
]

# Matplotlib 中文字体设置（按顺序尝试）
rcParams["font.sans-serif"] = [
    "Microsoft YaHei",  # Windows：微软雅黑
    "SimHei",  # 黑体
    "Songti SC",  # macOS：宋体
    "Arial Unicode MS",  # 通用 Unicode 字体
]
rcParams["axes.unicode_minus"] = False  # 解决坐标轴负号显示为方块的问题


# ======================
# 工具函数
# ======================


def _detect_group_cols(df: pd.DataFrame) -> List[str]:
    """自动检测 D 步使用的分组字段（存在的字段才用）。"""
    return [c for c in GROUP_KEY_CANDIDATES if c in df.columns]


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读取 C1 和 D 步输出数据，并标准化 ts 类型。"""
    if not PRELIM_PATH.exists():
        raise FileNotFoundError(f"未找到 C1 输出文件：{PRELIM_PATH}")
    if not FILLED_PATH.exists():
        raise FileNotFoundError(f"未找到 D 步输出文件：{FILLED_PATH}")

    print(f"📥 读取 C1 数据：{PRELIM_PATH}")
    df_pre = pd.read_csv(PRELIM_PATH)

    print(f"📥 读取 D 步补全数据：{FILLED_PATH}")
    df_fill = pd.read_csv(FILLED_PATH)

    df_pre["ts"] = pd.to_datetime(df_pre["ts"])
    df_fill["ts"] = pd.to_datetime(df_fill["ts"])

    return df_pre, df_fill


def compute_global_stats(df_pre: pd.DataFrame, df_fill: pd.DataFrame) -> dict:
    """全局统计：补前补后对比 + 补全行数量等。"""
    n_pre = len(df_pre)
    n_fill = len(df_fill)

    n_synth = int(df_fill.get("is_synthetic_row", pd.Series(False)).sum())
    n_price_filled = int(df_fill.get("is_filled_retail_price", pd.Series(False)).sum())
    n_volume_filled = int(df_fill.get("is_filled_volume", pd.Series(False)).sum())

    stats = {
        "n_pre": n_pre,
        "n_fill": n_fill,
        "n_synth": n_synth,
        "n_price_filled": n_price_filled,
        "n_volume_filled": n_volume_filled,
        "ratio_synth": n_synth / n_fill if n_fill > 0 else 0.0,
    }
    return stats


def compute_group_synth_stats(df_fill: pd.DataFrame) -> pd.DataFrame:
    """
    按 D 步的分组粒度，统计每个时间序列的补全占比。

    返回字段包括：
        group_size, synth_count, synth_ratio
    """
    if "is_synthetic_row" not in df_fill.columns:
        return pd.DataFrame()

    group_cols = _detect_group_cols(df_fill)
    if not group_cols:
        return pd.DataFrame()

    grp = df_fill.groupby(group_cols, dropna=False)
    agg = grp["is_synthetic_row"].agg(
        synth_count="sum",
        group_size="count",
    )
    agg["synth_ratio"] = agg["synth_count"] / agg["group_size"]
    agg = agg.sort_values("synth_ratio", ascending=False)
    return agg.reset_index()


def compute_daily_synth_stats(df_fill: pd.DataFrame) -> pd.DataFrame:
    """
    统计按日期的补全情况：
        - daily_synth_count：每天新增补全记录数
        - daily_total_count：每天总记录数
        - daily_synth_ratio：每日补全占比
    """
    if "is_synthetic_row" not in df_fill.columns:
        return pd.DataFrame()

    df = df_fill.copy()
    df["date"] = df["ts"].dt.normalize()

    grp = df.groupby("date", dropna=False)
    stats = grp["is_synthetic_row"].agg(
        daily_synth_count="sum",
        daily_total_count="count",
    )
    stats["daily_synth_ratio"] = stats["daily_synth_count"] / stats["daily_total_count"]
    stats = stats.sort_index()
    return stats.reset_index()


def add_text_page(pdf: PdfPages, title: str, lines: List[str]):
    """在 PDF 中添加一页纯文字（用于总览统计）。"""
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 纵向
    ax.axis("off")

    y = 0.95
    ax.text(
        0.5,
        y,
        title,
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
    )
    y -= 0.06

    for line in lines:
        ax.text(
            0.06,
            y,
            line,
            ha="left",
            va="top",
            fontsize=11,
            wrap=True,
        )
        y -= 0.035

    pdf.savefig(fig)
    plt.close(fig)


def add_group_synth_hist(pdf: PdfPages, group_stats: pd.DataFrame):
    """在 PDF 中添加一页：分组补全占比分布直方图。"""
    if group_stats.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        group_stats["synth_ratio"],
        bins=30,
        edgecolor="black",
    )
    ax.set_title("各时间序列补全占比分布（synth_ratio）")
    ax.set_xlabel("补全占比（synth_ratio）")
    ax.set_ylabel("时间序列数量")

    pdf.savefig(fig)
    plt.close(fig)


def add_daily_synth_timeseries(pdf: PdfPages, daily_stats: pd.DataFrame):
    """
    添加一页：按日期统计每天新增补全记录数量 的折线图。
    同时在右侧 Y 轴展示每日补全占比。
    """
    if daily_stats.empty:
        return

    fig, ax1 = plt.subplots(figsize=(10, 4))

    x = daily_stats["date"]
    y_count = daily_stats["daily_synth_count"]
    y_ratio = daily_stats["daily_synth_ratio"]

    # 左轴：每天补全记录数
    ax1.plot(x, y_count, "-o", markersize=3)
    ax1.set_title("按日期统计每天新增补全记录数量")
    ax1.set_xlabel("日期")
    ax1.set_ylabel("每日新增补全记录数")

    # 右轴：补全占比
    ax2 = ax1.twinx()
    ax2.plot(x, y_ratio, "--", linewidth=1)
    ax2.set_ylabel("每日补全占比")

    fig.autofmt_xdate()
    pdf.savefig(fig)
    plt.close(fig)


def add_example_series_plots(
    pdf: PdfPages,
    df_fill: pd.DataFrame,
    max_series: int = 8,
):
    """
    选取若干代表性序列画图：
    - 曲线：零售价 retail_price
    - 蓝点：原始日期（is_synthetic_row=False）
    - 红点：补全日期（is_synthetic_row=True）
    """
    if "is_synthetic_row" not in df_fill.columns:
        return

    group_cols = _detect_group_cols(df_fill)
    if not group_cols:
        return

    grp_stats = compute_group_synth_stats(df_fill)
    if grp_stats.empty:
        return

    # 取前 max_series 个补全占比较高的时间序列
    example_groups = grp_stats.head(max_series)

    for _, row in example_groups.iterrows():
        cond = []
        for col in group_cols:
            cond.append(df_fill[col].eq(row[col]))
        mask = cond[0]
        for c in cond[1:]:
            mask &= c

        g = df_fill[mask].copy()
        if g.empty:
            continue

        g = g.sort_values("ts")
        g["ts"] = pd.to_datetime(g["ts"])

        fig, ax = plt.subplots(figsize=(10, 4))

        # 主线：零售价
        ax.plot(
            g["ts"],
            g["retail_price"],
            "-",
            label="零售价（含补全）",
        )

        real = g[g["is_synthetic_row"] == False]
        synth = g[g["is_synthetic_row"] == True]

        # 原始记录
        ax.scatter(
            real["ts"],
            real["retail_price"],
            s=10,
            label="原始记录",
        )
        # 补全记录
        ax.scatter(
            synth["ts"],
            synth["retail_price"],
            s=20,
            marker="o",
            label="补全记录",
            c="red",
        )

        title_parts = [f"{col}={row[col]}" for col in group_cols]
        title = " | ".join(title_parts)
        ax.set_title(f"代表性时间序列价格走势：{title}")
        ax.set_xlabel("日期 ts")
        ax.set_ylabel("零售价 retail_price")
        ax.legend()

        fig.autofmt_xdate()
        pdf.savefig(fig)
        plt.close(fig)


# ======================
# 主流程
# ======================


def main():
    # 1. 加载数据
    df_pre, df_fill = load_data()

    # 2. 计算全局统计
    global_stats = compute_global_stats(df_pre, df_fill)

    # 3. 分组补全占比
    group_stats = compute_group_synth_stats(df_fill)

    # 4. 每日补全统计
    daily_stats = compute_daily_synth_stats(df_fill)

    # 5. 生成 PDF 报告
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"📝 正在生成 PDF 报告：{REPORT_PATH}")
    with PdfPages(REPORT_PATH) as pdf:
        # 页 1：整体文字总览
        lines = [
            "一、整体数据量对比：",
            f"  - C1 初步清洗后数据量：{global_stats['n_pre']:,} 行",
            f"  - D 步补全后数据量：{global_stats['n_fill']:,} 行",
            f"  - D 步新增补全记录数（is_synthetic_row=True）：{global_stats['n_synth']:,} 行",
            f"  - 补全记录占比：{global_stats['ratio_synth']:.2%}",
            "",
            "二、数值字段补全情况：",
            f"  - 零售价被填补次数（is_filled_retail_price=True）：{global_stats['n_price_filled']:,}",
            f"  - 成交量被填补次数（is_filled_volume=True）：{global_stats['n_volume_filled']:,}",
            "",
            "三、时间序列粒度说明：",
            "  - 每一条时间序列对应一个 (product_id, variety, spec, grade, shop_name, classify_name, color) 组合。",
            "  - 对每条时间序列统计：",
            "      synth_count = 该序列中 is_synthetic_row=True 的记录数；",
            "      group_size  = 序列总行数；",
            "      synth_ratio = synth_count / group_size。",
            "",
            "后续页面包括：",
            "  - 各时间序列补全占比分布直方图；",
            "  - 按日期统计每天新增补全记录数量的折线图（含每日补全占比）；",
            "  - 若干补全比例较高的代表性时间序列价格走势（蓝点为原始记录，红点为 D 步补全记录）。",
        ]
        add_text_page(pdf, "D 步：整天缺失补全 质量评估报告", lines)

        # 页 2：分组补全占比分布
        add_group_synth_hist(pdf, group_stats)

        # 页 3：按日期的补全数量折线图
        add_daily_synth_timeseries(pdf, daily_stats)

        # 页 4+：代表性序列价格走势
        add_example_series_plots(pdf, df_fill, max_series=8)

    print(f"✅ PDF 报告已生成：{REPORT_PATH}")


if __name__ == "__main__":
    main()
