# -*- coding: utf-8 -*-
"""
成交量预测模型评估（方案 A vs 方案 B）

- 读取：
    data/output/volume_prediction_result_A.csv
    data/output/volume_prediction_result_B.csv
- 自动适配：
    A：竖表（horizon, y_true, y_pred）
    B：宽表（y_volume_1d, pred_volume_1d_B, ...）
- 输出：
    reports/volume_model_eval_report.pdf

PDF 内容：
1）概览页：H1/H2/H3 的 N / MAPE / RMSE / R2 对比
2）每个 horizon：
    a. A vs B 散点图（真实值 vs 预测值）
    b. A vs B 误差分布（APE 直方图）
    c. 时间序列趋势对比（按天聚合）
    d. 按成交量分桶的 MAPE 条形图
    e. 按商品分组的 TopN 误差榜（A 相对 B 的提升）
3）结论与业务建议页
4）评估方法与口径说明页
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


A_PATH = Path("data/output/volume_prediction_result_A.csv")
B_PATH = Path("data/output/volume_prediction_result_B.csv")
REPORT_PATH = Path("reports/volume_model_eval_report.pdf")


# ---------- 基础工具 ----------


def setup_chinese_font():
    """尽量使用中文字体，防止乱码。"""
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算 N / MAPE / RMSE / R2，并保留 APE 序列用于后续画图。"""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"N": 0, "MAPE": np.nan, "RMSE": np.nan, "R2": np.nan, "APE": np.array([])}

    ape = np.abs((y_pred - y_true) / (y_true + 1e-6)) * 100
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

    # R2（拟合优度）
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    return {
        "N": int(len(y_true)),
        "MAPE": float(np.mean(ape)),
        "RMSE": float(rmse),
        "R2": float(r2),
        "APE": ape,
    }


# ---------- 加载与格式适配 ----------


def load_A() -> dict:
    """
    A 文件为竖表结构：
    ts, product_id, variety, ..., horizon, y_true, y_pred, abs_error, ape(%)

    返回：
        {1: {"ts":..., "product_id":..., "y_true":..., "y_pred":...}, ...}
    """
    if not A_PATH.exists():
        raise FileNotFoundError(f"未找到 A 方案结果文件：{A_PATH}")

    df = pd.read_csv(A_PATH)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])

    if "horizon" not in df.columns or "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError("A 文件缺少 horizon / y_true / y_pred 字段，请检查 model_predict_A 的输出格式。")

    if "product_id" not in df.columns:
        df["product_id"] = pd.NA

    data = {}
    for h in [1, 2, 3]:
        sub = df[df["horizon"] == h].copy()
        sub = sub.sort_values("ts")
        data[h] = {
            "ts": sub["ts"].values,
            "product_id": sub["product_id"].values,
            "y_true": sub["y_true"].values,
            "y_pred": sub["y_pred"].values,
        }
    return data


def load_B() -> dict:
    """
    B 文件为宽表结构：
    ts, product_id, variety, ..., y_volume_1d/2d/3d, pred_volume_1d/2d/3d_B
    """
    if not B_PATH.exists():
        raise FileNotFoundError(f"未找到 B 方案结果文件：{B_PATH}")

    df = pd.read_csv(B_PATH)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])

    if "product_id" not in df.columns:
        df["product_id"] = pd.NA

    data = {}
    for h in [1, 2, 3]:
        true_col = f"y_volume_{h}d"
        pred_col = f"pred_volume_{h}d_B"
        if true_col not in df.columns or pred_col not in df.columns:
            raise ValueError(f"B 文件缺少字段：{true_col} 或 {pred_col}")

        sub = df.sort_values("ts")
        data[h] = {
            "ts": sub["ts"].values,
            "product_id": sub["product_id"].values,
            "y_true": sub[true_col].values,
            "y_pred": sub[pred_col].values,
        }
    return data


# ---------- 概览页 & 基础可视化 ----------


def plot_summary_page(metricsA, metricsB, pdf: PdfPages):
    """概览页：H1/H2/H3 的 N / MAPE / RMSE / R2 对比。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    title = "成交量预测模型评估（方案 A vs 方案 B）"
    ax.text(0.02, 0.95, title, fontsize=18, weight="bold", va="top")

    lines = ["各预测期（H1/H2/H3）整体指标对比：", ""]
    for h in [1, 2, 3]:
        mA, mB = metricsA[h], metricsB[h]
        line = (
            f"H{h} | "
            f"A：N={mA['N']}, MAPE={mA['MAPE']:.2f}%, RMSE={mA['RMSE']:.1f}, R2={mA['R2']:.3f}  |  "
            f"B：N={mB['N']}, MAPE={mB['MAPE']:.2f}%, RMSE={mB['RMSE']:.1f}, R2={mB['R2']:.3f}"
        )
        lines.append(line)

    ax.text(0.02, 0.80, "\n".join(lines), fontsize=12, va="top")

    pdf.savefig(fig)
    plt.close(fig)


def plot_scatter_page(h, dataA_h, dataB_h, pdf: PdfPages):
    """散点图：真实值 vs 预测值（A & B）。"""
    y_true_A = dataA_h["y_true"].astype(float)
    y_pred_A = dataA_h["y_pred"].astype(float)
    y_true_B = dataB_h["y_true"].astype(float)
    y_pred_B = dataB_h["y_pred"].astype(float)

    all_true = np.concatenate([y_true_A, y_true_B])
    all_true = all_true[~np.isnan(all_true)]
    if len(all_true) == 0:
        return

    # 避免极端值影响，截断在 99 分位数
    vmax = np.nanpercentile(all_true, 99)
    if vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        np.clip(y_true_A, 0, vmax),
        np.clip(y_pred_A, 0, vmax),
        s=5,
        alpha=0.4,
        label="A：使用预测价格",
    )
    ax.scatter(
        np.clip(y_true_B, 0, vmax),
        np.clip(y_pred_B, 0, vmax),
        s=5,
        alpha=0.4,
        marker="x",
        label="B：仅历史价格",
    )

    ax.plot([0, vmax], [0, vmax], linestyle="--", linewidth=1, label="y = x 参考线")
    ax.set_xlabel("真实成交量")
    ax.set_ylabel("预测成交量")
    ax.set_title(f"H{h}：真实 vs 预测（散点图）")
    ax.set_xlim(0, vmax)
    ax.set_ylim(0, vmax)
    ax.legend(loc="upper left", fontsize=8)

    pdf.savefig(fig)
    plt.close(fig)


def plot_error_hist_page(h, mA, mB, pdf: PdfPages):
    """误差分布页：APE 直方图 + 分位数对比。"""
    ape_A = mA["APE"]
    ape_B = mB["APE"]

    if len(ape_A) == 0 or len(ape_B) == 0:
        return

    # 截断 0~200%，避免极端长尾
    ape_A_clip = np.clip(ape_A, 0, 200)
    ape_B_clip = np.clip(ape_B, 0, 200)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 200, 41)

    ax.hist(ape_A_clip, bins=bins, alpha=0.5, label="A：使用预测价格")
    ax.hist(ape_B_clip, bins=bins, alpha=0.5, label="B：仅历史价格")

    p90_A = np.percentile(ape_A, 90)
    p90_B = np.percentile(ape_B, 90)
    p95_A = np.percentile(ape_A, 95)
    p95_B = np.percentile(ape_B, 95)

    ax.axvline(p90_A, linestyle="--", linewidth=1, label=f"A P90={p90_A:.1f}%")
    ax.axvline(p90_B, linestyle="--", linewidth=1, label=f"B P90={p90_B:.1f}%")
    ax.axvline(p95_A, linestyle=":", linewidth=1, label=f"A P95={p95_A:.1f}%")
    ax.axvline(p95_B, linestyle=":", linewidth=1, label=f"B P95={p95_B:.1f}%")

    ax.set_xlim(0, 200)
    ax.set_xlabel("APE（绝对百分比误差 %）")
    ax.set_ylabel("样本数")
    ax.set_title(f"H{h}：误差分布（APE 直方图）")
    ax.legend(fontsize=8)

    pdf.savefig(fig)
    plt.close(fig)


# ---------- 公共：A/B 对齐 ----------


def build_aligned_df(dataA_h, dataB_h) -> pd.DataFrame:
    """
    将 A/B 两个方案在当前 horizon 上，按 (ts, product_id) 对齐成同一张表：
        ts, product_id, y_true, y_pred_A, y_pred_B

    只保留两边都存在的记录，保证后续 A/B 对比时长度一致。
    """
    dfA = pd.DataFrame(
        {
            "ts": dataA_h["ts"],
            "product_id": dataA_h["product_id"],
            "y_true": dataA_h["y_true"].astype(float),
            "y_pred_A": dataA_h["y_pred"].astype(float),
        }
    )
    dfB = pd.DataFrame(
        {
            "ts": dataB_h["ts"],
            "product_id": dataB_h["product_id"],
            "y_true_B": dataB_h["y_true"].astype(float),
            "y_pred_B": dataB_h["y_pred"].astype(float),
        }
    )

    merged = pd.merge(
        dfA,
        dfB[["ts", "product_id", "y_true_B", "y_pred_B"]],
        on=["ts", "product_id"],
        how="inner",
    )

    if merged.empty:
        return merged

    # 如有需要可以在此处新增 y_true 与 y_true_B 的一致性检查
    return merged


# ---------- 时间序列趋势 & 分桶分析 ----------


def plot_trend_page(h, dataA_h, dataB_h, pdf: PdfPages):
    """
    时间序列趋势对比：
    - 按 ts 聚合（按天求和）
    - 比较 真实量 vs A预测 vs B预测
    """
    dfA = pd.DataFrame(
        {
            "ts": dataA_h["ts"],
            "y_true": dataA_h["y_true"].astype(float),
            "y_pred_A": dataA_h["y_pred"].astype(float),
        }
    )
    dfB = pd.DataFrame(
        {
            "ts": dataB_h["ts"],
            "y_true_B": dataB_h["y_true"].astype(float),
            "y_pred_B": dataB_h["y_pred"].astype(float),
        }
    )

    if dfA.empty or dfB.empty:
        return

    # 按日期聚合（求和）
    aggA = dfA.groupby("ts").sum(numeric_only=True).reset_index()
    aggB = dfB.groupby("ts").sum(numeric_only=True).reset_index()

    # 对齐日期
    merged = pd.merge(aggA, aggB, on="ts", how="inner")
    if merged.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(merged["ts"], merged["y_true"], label="真实成交量（A视角）", linewidth=1.0)
    ax.plot(merged["ts"], merged["y_pred_A"], label="A 预测", linewidth=1.0)
    ax.plot(merged["ts"], merged["y_pred_B"], label="B 预测", linewidth=1.0, linestyle="--")

    ax.set_title(f"H{h}：按天聚合的成交量趋势（真实 vs A/B 预测）")
    ax.set_xlabel("日期")
    ax.set_ylabel("日成交量（聚合）")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()

    pdf.savefig(fig)
    plt.close(fig)


def plot_bucket_page(h, dataA_h, dataB_h, pdf: PdfPages):
    """
    按真实成交量分桶的 MAPE，对比 A / B。
    分桶示例：0~P25, P25~P50, P50~P75, P75~max

    注意：先对齐 A/B 的样本，再做分桶分析。
    """
    merged = build_aligned_df(dataA_h, dataB_h)
    if merged.empty:
        return

    y_true = merged["y_true"].values
    y_pred_A = merged["y_pred_A"].values
    y_pred_B = merged["y_pred_B"].values

    mask = ~np.isnan(y_true) & (y_true >= 0)
    y_true = y_true[mask]
    y_pred_A = y_pred_A[mask]
    y_pred_B = y_pred_B[mask]

    if len(y_true) == 0:
        return

    # 分位数阈值
    q25, q50, q75 = np.percentile(y_true, [25, 50, 75])
    bins = [0, q25, q50, q75, np.max(y_true) + 1e-6]
    labels = ["低（<=P25）", "中低（P25~P50）", "中高（P50~P75）", "高（>P75）"]

    bucket_idx = np.digitize(y_true, bins, right=True) - 1  # 0~3
    mape_A_list = []
    mape_B_list = []
    bucket_names = []

    for i, lab in enumerate(labels):
        mask_i = bucket_idx == i
        if not np.any(mask_i):
            continue
        yt = y_true[mask_i]
        ya = y_pred_A[mask_i]
        yb = y_pred_B[mask_i]

        ape_A = np.abs((ya - yt) / (yt + 1e-6)) * 100
        ape_B = np.abs((yb - yt) / (yt + 1e-6)) * 100

        mape_A_list.append(np.mean(ape_A))
        mape_B_list.append(np.mean(ape_B))
        bucket_names.append(lab)

    if not bucket_names:
        return

    x = np.arange(len(bucket_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, mape_A_list, width, label="A：使用预测价格")
    ax.bar(x + width / 2, mape_B_list, width, label="B：仅历史价格")

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_names)
    ax.set_ylabel("MAPE（%）")
    ax.set_title(f"H{h}：按成交量分桶的 MAPE 对比（基于对齐样本）")
    ax.legend(fontsize=8)

    pdf.savefig(fig)
    plt.close(fig)


# ---------- 按商品分组 TopN 误差榜 ----------


def plot_top_products_page(
    h, dataA_h, dataB_h, pdf: PdfPages, min_samples: int = 30, top_n: int = 10
):
    """
    按 product_id 维度聚合，统计：
        N, MAPE_A, MAPE_B, diff(B - A)
    并展示 A 相对 B 提升最大的 TopN 商品。

    注意：基于 A/B 对齐后的样本，避免长度不一致。
    """
    merged = build_aligned_df(dataA_h, dataB_h)
    if merged.empty:
        return

    df = merged[["product_id", "y_true", "y_pred_A", "y_pred_B"]].copy()
    df = df.dropna(subset=["y_true"])
    if df.empty:
        return

    def _agg(group: pd.DataFrame):
        yt = group["y_true"].values
        ya = group["y_pred_A"].values
        yb = group["y_pred_B"].values

        mask = ~np.isnan(yt)
        yt, ya, yb = yt[mask], ya[mask], yb[mask]
        if len(yt) == 0:
            return pd.Series({"N": 0, "MAPE_A": np.nan, "MAPE_B": np.nan, "DIFF": np.nan})

        ape_A = np.abs((ya - yt) / (yt + 1e-6)) * 100
        ape_B = np.abs((yb - yt) / (yt + 1e-6)) * 100
        mape_A = np.mean(ape_A)
        mape_B = np.mean(ape_B)
        diff = mape_B - mape_A  # diff > 0 表示 A 好于 B

        return pd.Series({"N": len(yt), "MAPE_A": mape_A, "MAPE_B": mape_B, "DIFF": diff})

    grouped = (
        df.groupby("product_id")[["y_true", "y_pred_A", "y_pred_B"]]
        .apply(_agg)
        .reset_index()
    )

    # 过滤样本数过少的商品，防止偶然值
    grouped = grouped[grouped["N"] >= min_samples]
    grouped = grouped.dropna(subset=["MAPE_A", "MAPE_B", "DIFF"])

    if grouped.empty:
        return

    # 选取 A 相对 B 提升最大的 TopN
    grouped = grouped.sort_values("DIFF", ascending=False).head(top_n)

    # 准备表格数据
    table_data = []
    for _, row in grouped.iterrows():
        pid = str(row["product_id"])
        table_data.append(
            [
                pid,
                int(row["N"]),
                f"{row['MAPE_A']:.1f}",
                f"{row['MAPE_B']:.1f}",
                f"{row['DIFF']:.1f}",
            ]
        )

    col_labels = ["product_id", "样本数", "MAPE_A(%)", "MAPE_B(%)", "B-A 差值"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.set_title(f"H{h}：按商品维度的 Top{len(table_data)} 误差对比（A 相对 B 提升）", pad=20)

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    note = (
        "说明：\n"
        f"  • 仅展示样本数 ≥ {min_samples} 的商品；\n"
        "  • “B-A 差值”>0 表示 A 的 MAPE 低于 B，A 表现更好；\n"
        "  • 可据此筛选关键品类，做专项模型或业务规则优化。"
    )
    ax.text(0.02, 0.02, note, fontsize=8, va="bottom", ha="left")

    pdf.savefig(fig)
    plt.close(fig)


# ---------- 结论 & 方法说明 ----------


def plot_conclusion_page(metricsA, metricsB, pdf: PdfPages):
    """生成一个文字结论页。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    lines = []
    lines.append("综合评估结论（成交量预测 A vs B）")
    lines.append("")

    for h in [1, 2, 3]:
        mA, mB = metricsA[h], metricsB[h]
        better = "A" if mA["MAPE"] <= mB["MAPE"] else "B"
        lines.append(
            f"● H{h}：方案 {better} 在总体 MAPE 上更优 "
            f"(A={mA['MAPE']:.2f}% / B={mB['MAPE']:.2f}%)，"
            f"且 R2 分别为 A={mA['R2']:.3f} / B={mB['R2']:.3f}。"
        )

    lines.append("")
    lines.append("业务向导性解读示例：")
    lines.append(
        "1）若希望在大部分普通交易量档位下获得更稳的预测，可优先采用整体 MAPE 更低的方案；"
    )
    lines.append(
        "2）若在高成交量或重点品种上对误差更敏感，可结合“分桶 MAPE 图”重点检查高销量桶的表现；"
    )
    lines.append(
        "3）方案 A 引入了“预测价格”这一前视特征，在价格波动较大、量价联动明显的场景下，通常会比方案 B 更有优势；"
    )
    lines.append(
        "4）在实际落地时，可以在系统中同时保留 A/B 两个版本，通过线上 A/B Test 或滚动窗口监控误差，持续校正。"
    )

    ax.text(0.03, 0.95, lines[0], fontsize=18, weight="bold", va="top")
    ax.text(0.03, 0.88, "\n".join(lines[2:5]), fontsize=12, va="top")
    ax.text(0.03, 0.65, "\n".join(lines[6:]), fontsize=11, va="top")

    pdf.savefig(fig)
    plt.close(fig)


def plot_methodology_page(pdf: PdfPages):
    """方法与口径说明页，方便对外展示时说明评估逻辑。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    title = "评估方法与口径说明"
    lines = [
        "1）样本范围：",
        "   - 使用统一时间切分规则：约 80% 时间序列用于训练，20% 用于验证；",
        "   - A/B 方案采用相同的验证集时间段，确保可比性；",
        "",
        "2）指标定义：",
        "   - MAPE：平均绝对百分比误差，用于衡量整体相对误差水平；",
        "   - RMSE：均方根误差，更关注大误差样本的影响；",
        "   - R2：拟合优度，越接近 1 表示模型对趋势解释能力越强；",
        "",
        "3）分桶与分组：",
        "   - 成交量分桶：按照真实成交量的四分位数，将样本分为低/中低/中高/高四档；",
        "   - 商品分组：按 product_id 聚合，仅展示样本数较多（如 ≥30）的商品；",
        "",
        "4）注意事项：",
        "   - 报告中所有结论均基于当前历史数据，建议定期滚动更新；",
        "   - 在节假日、极端行情等特殊场景下，可结合业务规则进行人工干预；",
        "   - 可进一步增加节假日/周几/活动标签等特征，提升关键时段的预测稳定性。",
    ]

    ax.text(0.03, 0.95, title, fontsize=18, weight="bold", va="top")
    ax.text(0.03, 0.88, "\n".join(lines), fontsize=11, va="top")

    pdf.savefig(fig)
    plt.close(fig)


# ---------- 主流程 ----------


def main():
    setup_chinese_font()

    print("📥 加载 A 方案预测结果 ...")
    dataA = load_A()

    print("📥 加载 B 方案预测结果 ...")
    dataB = load_B()

    print("📊 计算 A / B 指标 ...")
    metricsA = {}
    metricsB = {}
    for h in [1, 2, 3]:
        mA = compute_metrics(
            dataA[h]["y_true"].astype(float),
            dataA[h]["y_pred"].astype(float),
        )
        mB = compute_metrics(
            dataB[h]["y_true"].astype(float),
            dataB[h]["y_pred"].astype(float),
        )
        metricsA[h] = mA
        metricsB[h] = mB
        print(
            f"H{h} → "
            f"A: N={mA['N']}, MAPE={mA['MAPE']:.2f}%, RMSE={mA['RMSE']:.1f}, R2={mA['R2']:.3f} | "
            f"B: N={mB['N']}, MAPE={mB['MAPE']:.2f}%, RMSE={mB['RMSE']:.1f}, R2={mB['R2']:.3f}"
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n📝 生成 PDF 报告：{REPORT_PATH}")
    with PdfPages(REPORT_PATH) as pdf:
        # 1. 总览页
        plot_summary_page(metricsA, metricsB, pdf)

        # 2. 每个 horizon 的详细分析
        for h in [1, 2, 3]:
            # 散点图
            plot_scatter_page(h, dataA[h], dataB[h], pdf)
            # 误差分布
            plot_error_hist_page(h, metricsA[h], metricsB[h], pdf)
            # 时间序列趋势
            plot_trend_page(h, dataA[h], dataB[h], pdf)
            # 分桶分析
            plot_bucket_page(h, dataA[h], dataB[h], pdf)
            # 商品维度 TopN 误差榜
            plot_top_products_page(h, dataA[h], dataB[h], pdf)

        # 3. 结论页
        plot_conclusion_page(metricsA, metricsB, pdf)

        # 4. 方法说明页
        plot_methodology_page(pdf)

    print("✅ 成交量预测评估报告生成完成。")


if __name__ == "__main__":
    main()
