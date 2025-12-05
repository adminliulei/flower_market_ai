import os

print("📌 开始训练成交量预测模型（方案 A：未来预测价格）...")
os.system("python -m src.prediction_models.volume_pred.model_train_A")

print("\n📌 开始训练成交量预测模型（方案 B：历史价格）...")
os.system("python -m src.prediction_models.volume_pred.model_train_B")

print("\n🎉 成交量预测模型 A + B 全部训练完成！")
