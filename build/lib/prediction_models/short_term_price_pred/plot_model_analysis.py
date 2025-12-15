# src/prediction_models/short_term_price_pred/plot_model_analysis.py
"""
模型分析与可视化脚本（基于已训练模型和预测结果）
功能：
    - 读取模型 + 元数据
    - 读取 price_prediction_result.csv
    - 按 horizon 分组生成图表
    - 支持 1d, 2d, 3d
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from pathlib import Path
import joblib

# -------------------------
# 路径配置
# -------------------------

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models" / "artifacts" / "price_model_v1"
DATA_OUTPUT_DIR = ROOT / "data" / "output"

VISUALIZATION_DIR = ROOT / "reports" / "model_visualization"
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# 1. 加载模型和元数据
# -------------------------

def load_model_and_metadata():
    """加载所有模型及其元数据"""
    model_paths = {
        '1d': MODEL_DIR / "model_1d.pkl",
        '2d': MODEL_DIR / "model_2d.pkl",
        '3d': MODEL_DIR / "model_3d.pkl"
    }
    metadata_paths = {
        '1d': MODEL_DIR / "metadata_1d.json",
        '2d': MODEL_DIR / "metadata_2d.json",
        '3d': MODEL_DIR / "metadata_3d.json"
    }

    models = {}
    metadata = {}

    for h in ['1d', '2d', '3d']:
        if model_paths[h].exists() and metadata_paths[h].exists():
            models[h] = joblib.load(model_paths[h])
            with open(metadata_paths[h], 'r', encoding='utf-8') as f:
                metadata[h] = json.load(f)
        else:
            print(f"⚠️ 跳过 {h} 模型：文件不存在")

    return models, metadata


# -------------------------
# 2. 加载预测结果
# -------------------------

def load_prediction_results():
    """加载价格预测结果"""
    result_path = DATA_OUTPUT_DIR / "price_prediction_result.csv"
    if not result_path.exists():
        raise FileNotFoundError(f"预测结果文件不存在：{result_path}")

    df = pd.read_csv(result_path)
    df['ts'] = pd.to_datetime(df['ts'])
    return df


# -------------------------
# 3. 绘制所有图表（按 horizon 分组）
# -------------------------

def plot_all_visualizations(models, metadata, pred_df):
    """为每个 horizon 生成图表"""
    print("📊 开始生成模型可视化图表...")

    # 设置中文字体（可选）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # 支持的 horizon
    horizons = ['1d', '2d', '3d']

    for h in horizons:
        if h not in models:
            continue

        # 过滤当前 horizon 的数据
        df_h = pred_df[pred_df['horizon'] == f'1y_price_{h}']
        if df_h.empty:
            print(f"⚠️ {h} 数据为空，跳过")
            continue

        target_col = df_h['target_col'].iloc[0]
        print(f"\n📈 处理 {h} 模型（目标列：{target_col}）")

        # 1. 决策树结构图
        try:
            tree_digraph = lgb.create_tree_digraph(models[h], tree_index=0)
            tree_digraph.render(
                filename=VISUALIZATION_DIR / f"decision_tree_{h}",
                format="png",
                cleanup=True
            )
            print(f"✅ 已生成：{h} 决策树结构图")
        except Exception as e:
            print(f"❌ {h} 决策树生成失败：{e}")

        # 2. 预测 vs 真实值散点图
        try:
            plt.figure(figsize=(8, 6))
            plt.scatter(df_h['y_true'], df_h['y_pred'], alpha=0.6, s=10)
            plt.plot([df_h['y_true'].min(), df_h['y_true'].max()],
                     [df_h['y_true'].min(), df_h['y_true'].max()], 'r--', lw=2)
            plt.xlabel('True Price')
            plt.ylabel('Predicted Price')
            plt.title(f'Prediction vs True Value ({h})')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(VISUALIZATION_DIR / f"pred_vs_true_scatter_{h}.png")
            plt.close()
            print(f"✅ 已生成：{h} 散点图")
        except Exception as e:
            print(f"❌ {h} 散点图生成失败：{e}")

        # 3. 残差图
        try:
            residuals = df_h['y_true'] - df_h['y_pred']
            plt.figure(figsize=(8, 6))
            plt.scatter(df_h['y_pred'], residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Value')
            plt.ylabel('Residual (True - Pred)')
            plt.title(f'Residual Plot ({h})')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(VISUALIZATION_DIR / f"residual_plot_{h}.png")
            plt.close()
            print(f"✅ 已生成：{h} 残差图")
        except Exception as e:
            print(f"❌ {h} 残差图生成失败：{e}")

        # 4. 时间序列预测 vs 真实走势
        try:
            df_h_sorted = df_h.sort_values('ts').reset_index(drop=True)
            plt.figure(figsize=(12, 5))
            plt.plot(df_h_sorted['ts'], df_h_sorted['y_true'], label='True', alpha=0.8)
            plt.plot(df_h_sorted['ts'], df_h_sorted['y_pred'], label='Predicted', alpha=0.8)
            plt.title(f'Time Series: Prediction vs True ({h})')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(VISUALIZATION_DIR / f"time_series_prediction_{h}.png")
            plt.close()
            print(f"✅ 已生成：{h} 时间序列图")
        except Exception as e:
            print(f"❌ {h} 时间序列图生成失败：{e}")

        # 5. 特征重要性图
        try:
            plt.figure(figsize=(10, 6))
            lgb.plot_importance(models[h], importance_type='gain', max_num_features=20, height=0.8)
            plt.title(f'Feature Importance (Gain) - {h}')
            plt.tight_layout()
            plt.savefig(VISUALIZATION_DIR / f"feature_importance_gain_{h}.png")
            plt.close()
            print(f"✅ 已生成：{h} 特征重要性图")
        except Exception as e:
            print(f"❌ {h} 特征重要性图生成失败：{e}")


# -------------------------
# 主函数
# -------------------------

def main():
    try:
        models, metadata = load_model_and_metadata()
        pred_df = load_prediction_results()

        plot_all_visualizations(models, metadata, pred_df)

        print("\n🎉 所有图表已生成！")
        print(f"📁 存放路径：{VISUALIZATION_DIR}")
    except Exception as e:
        print(f"❌ 运行失败：{e}")


if __name__ == "__main__":
    main()