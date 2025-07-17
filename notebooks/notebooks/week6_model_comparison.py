# 加载预测结果
results = []

# 遍历每家医院
for branch in branches:
    # 拿出 Prophet 与 XGBoost 的预测
    true_val = ...  # 真实值
    prophet_pred = ...  # Prophet 的预测
    xgb_pred = ...  # XGBoost 的预测
    
    # 计算误差
    prophet_mae = abs(true_val - prophet_pred)
    xgb_mae = abs(true_val - xgb_pred)

    # 选择最优模型
    if prophet_mae < xgb_mae:
        best_model = "Prophet"
        best_pred = prophet_pred
    else:
        best_model = "XGBoost"
        best_pred = xgb_pred

    # 存储结果
    results.append({
        "hospital_branch": branch,
        "true_value": true_val,
        "xgb_pred": xgb_pred,
        "prophet_pred": prophet_pred,
        "best_model": best_model,
        "best_pred": best_pred
    })

# 保存为 CSV
df_results = pd.DataFrame(results)
df_results.to_csv("data/Forecast_Results.csv", index=False)
print("✅ Forecast_Results.csv 已保存")
