# src/prediction_models/short_term_price_pred/plot_model_analysis.py
"""
模型分析与可视化脚本（基于已训练模型和预测结果）
适配 horizon = [1, 2, 3]（整数）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from pathlib import Path
import joblib
import json

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

def load_models():
    models = {}
    for h in [1, 2, 3]:
        model_path = MODEL_DIR / f"model_{h}d.pkl"
        if model_path.exists():
            models[h] = joblib.load(model_path)
        else:
            print(f"⚠️ 模型文件缺失：{model_path}")
    return models


# -------------------------
# 2. 加载预测结果
# -------------------------

def load_prediction_results():
    result_path = DATA_OUTPUT_DIR / "price_prediction_result.csv"
    if not result_path.exists():
        raise FileNotFoundError(f"预测结果文件不存在：{result_path}")

    df = pd.read_csv(result_path)
    df['ts'] = pd.to_datetime(df['ts'])

    # 打印 unique horizon 值用于调试
    print("🔍 可用的 horizon 值：", sorted(df['horizon'].unique()))
    return df


# -------------------------
# 3. 绘制所有图表
# -------------------------

def plot_all_visualizations(models, pred_df):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    for h in [1, 2, 3]:
        if h not in models:
            continue

        # 匹配 horizon = 1, 2, 3（整数）
        df_h = pred_df[pred_df['horizon'] == h]

        if df_h.empty:
            print(f"⚠️ {h}d 数据为空（未找到 horizon={h}）")
            continue

        print(f"\n📈 处理 {h}d 模型，共 {len(df_h)} 条记录")

        # --- 1. 决策树结构图 ---
        try:
            tree_digraph = lgb.create_tree_digraph(models[h], tree_index=0)
            tree_digraph.render(
                filename=VISUALIZATION_DIR / f"decision_tree_{h}d",
                format="png",
                cleanup=True
            )
            print(f"✅ {h}d: 决策树结构图")
        except Exception as e:
            print(f"❌ {h}d: 决策树失败 - {e}")

        # --- 2. 预测 vs 真实散点图 ---
        try:
            plt.figure(figsize=(8, 6))
            plt.scatter(df_h['y_true'], df_h['y_pred'], alpha=0.5, s=10)
            min_val, max_val = df_h[['y_true', 'y_pred']].min().min(), df_h[['y_true', 'y_pred']].max().max()
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            plt.xlabel('True Price')
            plt.ylabel('Predicted Price')
            plt.title(f'Prediction vs True ({h}d)')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(VISUALIZATION_DIR / f"pred_vs_true_scatter_{h}d.png")
            plt.close()
            print(f"✅ {h}d: 散点图")
        except Exception as e:
            print(f"❌ {h}d: 散点图失败 - {e}")

        # --- 3. 残差图 ---
        try:
            residuals = df_h['y_true'] - df_h['y_pred']
            plt.figure(figsize=(8, 6))
            plt.scatter(df_h['y_pred'], residuals, alpha=0.5)
            plt.axhline(0, color='r', linestyle='--')
            plt.xlabel('Predicted Price')
            plt.ylabel('Residual (True - Pred)')
            plt.title(f'Residual Plot ({h}d)')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(VISUALIZATION_DIR / f"residual_plot_{h}d.png")
            plt.close()
            print(f"✅ {h}d: 残差图")
        except Exception as e:
            print(f"❌ {h}d: 残差图失败 - {e}")

        # --- 4. 时间序列预测图 ---
        try:
            df_sorted = df_h.sort_values('ts').reset_index(drop=True)
            plt.figure(figsize=(12, 5))
            plt.plot(df_sorted['ts'], df_sorted['y_true'], label='True', alpha=0.8)
            plt.plot(df_sorted['ts'], df_sorted['y_pred'], label='Predicted', alpha=0.8)
            plt.title(f'Time Series Prediction ({h}d)')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(VISUALIZATION_DIR / f"time_series_prediction_{h}d.png")
            plt.close()
            print(f"✅ {h}d: 时间序列图")
        except Exception as e:
            print(f"❌ {h}d: 时间序列图失败 - {e}")

        # --- 5. 特征重要性图 ---
        try:
            plt.figure(figsize=(10, 6))
            lgb.plot_importance(models[h], importance_type='gain', max_num_features=20, height=0.8)
            plt.title(f'Feature Importance (Gain) - {h}d')
            plt.tight_layout()
            plt.savefig(VISUALIZATION_DIR / f"feature_importance_gain_{h}d.png")
            plt.close()
            print(f"✅ {h}d: 特征重要性图")
        except Exception as e:
            print(f"❌ {h}d: 特征重要性图失败 - {e}")


# -------------------------
# 主函数
# -------------------------

def main():
    try:
        models = load_models()
        if not models:
            print("❌ 未加载任何模型，退出")
            return

        pred_df = load_prediction_results()
        plot_all_visualizations(models, pred_df)

        print("\n🎉 所有图表已生成！")
        print(f"📁 存放路径：{VISUALIZATION_DIR}")
    except Exception as e:
        print(f"❌ 运行失败：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()