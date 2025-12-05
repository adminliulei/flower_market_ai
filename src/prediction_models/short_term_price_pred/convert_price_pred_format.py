import pandas as pd
from pathlib import Path

INPUT = Path("data/output/price_prediction_result.csv")
OUTPUT = Path("data/output/price_prediction_wide.csv")

def main():
    print(f"📥 读取价格预测结果：{INPUT}")
    df = pd.read_csv(INPUT)

    # 选择必要字段
    needed = ["ts", "product_id", "variety", "horizon", "y_pred"]
    existed = [c for c in needed if c in df.columns]
    df = df[existed]

    # pivot 成宽格式
    df_wide = df.pivot_table(
        index=["ts", "product_id", "variety"],
        columns="horizon",
        values="y_pred",
        aggfunc="first"
    ).reset_index()

    # 重命名列
    df_wide = df_wide.rename(columns={
        1: "pred_price_1d",
        2: "pred_price_2d",
        3: "pred_price_3d",
    })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df_wide.to_csv(OUTPUT, index=False)

    print(f"✅ 已生成宽格式预测文件：{OUTPUT}")
    print("   字段：ts, product_id, variety, pred_price_1d/2d/3d")

if __name__ == "__main__":
    main()
